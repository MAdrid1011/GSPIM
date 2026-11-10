#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using torch::Tensor;
#ifndef GSPIM_GENERATION_TRACE
#define GSPIM_GENERATION_TRACE false
#endif
#include <assert.h>
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value)::"memory");return value;
}
__device__ __forceinline__ void rotation_pair(const float *source, float *q0, float *q1,
                                float &omega, float &den) {
  float n0 = 0, n1 = 0, dot = 0;
  for (int c = 0; c < 4; ++c) {
    q0[c] = source[c];
    q1[c] = source[4 + c];
    n0 += q0[c] * q0[c];
    n1 += q1[c] * q1[c];
  }
  for (int c = 0; c < 4; ++c) {
    q0[c] *= rsqrtf(n0);
    q1[c] *= rsqrtf(n1);
    dot += q0[c] * q1[c];
  }
  omega = fmaxf(acosf(fminf(1 - 1e-4f, fmaxf(-1 + 1e-4f, dot))), 1e-4f);
  den = fmaxf(sinf(omega), 1e-4f);
}

template <bool Prepared = false, bool Trace = false>
__global__ void generate_kernel(const float *xyz, const float *disp,
                                const float *motion, const float *rot,
                                const float *mrot, const float *scale,
                                const float *opacity, const float *center,
                                const float *var, const float *times, int ns,
                                int nd, int K, int W, float duration,
                                float interval, float shift, float varpad,
                                float *means, float *cov, float *op,const float* camera,int step,float range,int mode,
float* depth,float* score,float* minimum,unsigned long long* publication,unsigned long long* loaded,const int64_t* selected,int selected_nd) {
  int g=blockIdx.x*blockDim.x+threadIdx.x,N=ns+selected_nd;
  int source_g=g<ns?g:(g<N?ns+selected[g-ns]:0);
  if(Trace&&!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }

  __shared__ volatile unsigned hashes[256];
  float position[3]={},velocity[3]={},quaternion[4]={},axes[3]={};
  float centers[2]={},variances[2]={},positions[15]={},rotations[12]={},pairs[4]={};
  float amplitude=0;
  int key0=(int)floorf((times[0]+shift)/interval);
  int last_key=(int)floorf((times[W-1]+shift)/interval),delta=last_key-key0;
  assert(Prepared&&delta>=0&&delta<=1);
  if(g<N) {
    amplitude=opacity[source_g];
    #pragma unroll
    for(int c=0;c<3;++c)axes[c]=scale[3*source_g+c];
    if(g<ns) {
      #pragma unroll
      for(int c=0;c<3;++c){position[c]=xyz[3*g+c];velocity[c]=disp[3*g+c];}
      #pragma unroll
      for(int c=0;c<4;++c)quaternion[c]=rot[4*g+c];
    } else {
      int j=selected[g-ns];
      #pragma unroll
      for(int c=0;c<2;++c){centers[c]=center[2*j+c];variances[c]=var[2*j+c];}
      #pragma unroll
      for(int k=0;k<5;++k)if(k<4+delta) {
        #pragma unroll
        for(int c=0;c<3;++c)positions[3*k+c]=motion[((key0-1+k)*3+c)*nd+j];
      }
      #pragma unroll
      for(int k=0;k<3;++k)if(k<2+delta) {
        #pragma unroll
        for(int c=0;c<4;++c)rotations[4*k+c]=mrot[(4*(key0+k)+c)*nd+j];
      }
      #pragma unroll
      for(int k=0;k<2;++k)if(k<=delta) {
        #pragma unroll
        for(int c=0;c<2;++c)pairs[2*k+c]=mrot[(4*K+2*(key0+k)+c)*nd+j];
      }
    }
    unsigned hash=__float_as_uint(amplitude);
    #pragma unroll
    for(int c=0;c<3;++c)hash^=__float_as_uint(axes[c])^__float_as_uint(position[c])^__float_as_uint(velocity[c]);
    #pragma unroll
    for(int c=0;c<4;++c)hash^=__float_as_uint(quaternion[c])^__float_as_uint(pairs[c]);
    #pragma unroll
    for(int c=0;c<2;++c)hash^=__float_as_uint(centers[c])^__float_as_uint(variances[c]);
    #pragma unroll
    for(int c=0;c<15;++c)hash^=__float_as_uint(positions[c]);
    #pragma unroll
    for(int c=0;c<12;++c)hash^=__float_as_uint(rotations[c]);
    hashes[threadIdx.x]=hash;
  }
  __syncthreads();
  if(Trace&&!threadIdx.x)loaded[blockIdx.x]=timestamp();
  if(g<N) {
    for(int f=0;f<(g<ns?1:W);++f) {
  int idx = f * N + g;
  float t = times[f];
  float q[4];
  if (g < ns) {
    for (int frame = 0; frame < W; ++frame) {
      for (int c = 0; c < 3; ++c)
        means[3 * (frame * N + g) + c] =
            position[c] + velocity[c] * (times[frame] / duration);
      op[frame * N + g] = amplitude;
    }
    for (int c = 0; c < 4; ++c)
      q[c] = quaternion[c];
  } else {
    int j = selected[g-ns];
    float u = (t + shift) / interval;
    int k = (int)floorf(u);
    float a = u - k;
    for (int c = 0; c < 3; ++c) {
      int base = Prepared ? (k * 3 + c) * nd + j : (j * K + k) * 3 + c;
      int step = Prepared ? 3 * nd : 3;
      float y0 = (k==key0?positions[c]:positions[3+c]), y1 = (k==key0?positions[3+c]:positions[6+c]), y2 = (k==key0?positions[6+c]:positions[9+c]),
            y3 = (k==key0?positions[9+c]:positions[12+c]);
      means[3 * idx + c] = (2 * a * a * a - 3 * a * a + 1) * y1 +
                           (a * a * a - 2 * a * a + a) * (y2 - y0) * .5f +
                           (-2 * a * a * a + 3 * a * a) * y2 +
                           (a * a * a - a * a) * (y3 - y1) * .5f;
    }
    float q0[4], q1[4], omega, den;
    if (Prepared) {
      const float *pair = mrot + j;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        q0[c] = (k==key0?rotations[c]:rotations[4+c]);
        q1[c] = (k==key0?rotations[4+c]:rotations[8+c]);
      }
      omega = (k==key0?pairs[0]:pairs[2]);
      den = (k==key0?pairs[1]:pairs[3]);
    } else {
      rotation_pair(mrot + (j * K + k) * 4, q0, q1, omega, den);
    }
    float p0 = sinf((1 - a) * omega) / den, p1 = sinf(a * omega) / den,
          sum = fmaxf(p0 + p1, 1e-4f), norm = 0, abs_sum = 0;
    for (int c = 0; c < 4; ++c) {
      q[c] = q0[c] * (p0 / sum) + q1[c] * (p1 / sum);
      norm += q[c] * q[c];
      abs_sum += fabsf(q[c]);
    }
    for (int c = 0; c < 4; ++c)
      q[c] = abs_sum > 1e-4f ? q[c] * rsqrtf(norm) : q0[c];
    float c0 = centers[0], c1 = centers[1], m = fminf(u - c0, u - c1),
          v = (u > c0 || u > c1) ? variances[1] : variances[0];
    float sigma = Prepared ? v : expf(v) + varpad / interval / 2.36f;
    op[idx] = amplitude *
              ((c0 - u) * (c1 - u) < 0 ? 1 : expf(-m * m / (sigma * sigma)));
  }
  float w = q[0], x = q[1], y = q[2], z = q[3];
  float R[9] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                2 * (x * z + w * y),     2 * (x * y + w * z),
                1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y),     2 * (y * z + w * x),
                1 - 2 * (x * x + y * y)};
  int n = 0;
  for (int a = 0; a < 3; ++a)
    for (int b = a; b < 3; ++b) {
      float v = 0;
      for (int k = 0; k < 3; ++k)
        v += R[3 * a + k] * R[3 * b + k] * axes[k] * axes[k];
      if (g < ns)
        for (int frame = 0; frame < W; ++frame)
          cov[6 * (frame * N + g) + n] = v;
      else
        cov[6 * idx + n] = v;
      ++n;
    }
    }
  float a=camera[0],b=camera[step],c=camera[2*step],t=camera[3*step];
  float previous=0,change=0,low=0;
  for(int f=0;f<W;++f) {
    float3 p=((const float3*)means)[f*N+g];float x=p.x,y=p.y,z=p.z;
    int order=mode/4;
    if(order==1){float tmp=x;x=z;z=tmp;tmp=a;a=c;c=tmp;}
    if(order==2){float tmp=x;x=y;y=tmp;tmp=a;a=b;b=tmp;}
    if(order==3){float tmp=y;y=z;z=tmp;tmp=b;b=c;c=tmp;}
    if(order==4){float tmp=x;x=y;y=z;z=tmp;tmp=a;a=b;b=c;c=tmp;}
    if(order==5){float tmp=x;x=z;z=y;y=tmp;tmp=a;a=c;c=b;b=tmp;}
    float d;
    if(mode%4==0)d=__fmaf_rn(z,c,__fmaf_rn(y,b,__fmul_rn(x,a)));
    else if(mode%4==1)d=__fadd_rn(__fadd_rn(__fmul_rn(x,a),__fmul_rn(y,b)),__fmul_rn(z,c));
    else if(mode%4==2)d=__fadd_rn(__fmaf_rn(y,b,__fmul_rn(x,a)),__fmul_rn(z,c));
    else d=__fmaf_rn(z,c,__fadd_rn(__fmul_rn(x,a),__fmul_rn(y,b)));
    d=__fadd_rn(d,t);depth[f*N+g]=d;
    if(!f)low=d;
    else{change+=fabsf(d-previous);low=fminf(low,d);}
    previous=d;
    // Restore camera order for the next frame.
    a=camera[0];b=camera[step];c=camera[2*step];
  }
  minimum[g]=low;
  float mean=W>1?__fmul_rn(change,1.f/(W-1)):0.f;
  score[g]=expf(__fmul_rn(-mean,1.f/range));
  }
  if(Trace) {
    __threadfence();__syncthreads();
    if(!threadIdx.x)publication[blockIdx.x*3+1]=timestamp();
  }
}
std::vector<Tensor> generate(Tensor xyz, Tensor disp, Tensor motion, Tensor rot,
                             Tensor mrot, Tensor scale, Tensor opacity,
                             Tensor center, Tensor var, Tensor times,
                             float duration, float interval, float shift,
                             float varpad,Tensor view,float range,int mode,Tensor selected) {
  c10::cuda::CUDAGuard guard(xyz.device());
  bool prepared = mrot.dim() == 2;
  int ns = xyz.size(0), nd = scale.size(0) - ns;
  int K = prepared ? motion.size(0) : motion.size(1);
  int W = times.numel(), selected_nd=selected.numel(), N=ns+selected_nd;
  auto means = torch::empty({W, N, 3}, xyz.options()),
       cov = torch::empty({W, N, 6}, xyz.options()),
       op = torch::empty({W, N}, xyz.options());
  TORCH_CHECK(prepared&&W>=1&&W<=5,"prepared window generation required");
  auto depth=torch::empty({W,N},xyz.options()),score=torch::empty({N},xyz.options()),minimum=torch::empty_like(score);
  auto publication=torch::empty({GSPIM_GENERATION_TRACE?(N+255)/256:0,3},xyz.options().dtype(torch::kInt64));
  auto loaded=torch::empty({GSPIM_GENERATION_TRACE?(N+255)/256:0},xyz.options().dtype(torch::kInt64));
  TORCH_CHECK(interval>=4,"prefetched generation requires adjacent interpolation keys");
  auto kernel=generate_kernel<true,GSPIM_GENERATION_TRACE>;
  kernel<<<(N + 255) / 256, 256, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
      xyz.data_ptr<float>(), disp.data_ptr<float>(), motion.data_ptr<float>(),
      rot.data_ptr<float>(), mrot.data_ptr<float>(), scale.data_ptr<float>(),
      opacity.data_ptr<float>(), center.data_ptr<float>(),
      var.data_ptr<float>(), times.data_ptr<float>(), ns, nd, K, W, duration,
      interval, shift, varpad, means.data_ptr<float>(), cov.data_ptr<float>(),
      op.data_ptr<float>(),view.data_ptr<float>()+2*view.stride(1),view.stride(2),range,mode,
      depth.data_ptr<float>(),score.data_ptr<float>(),minimum.data_ptr<float>(),
      (unsigned long long*)publication.data_ptr<int64_t>(),(unsigned long long*)loaded.data_ptr<int64_t>(),selected.data_ptr<int64_t>(),selected_nd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {means,cov,op,depth,score,minimum,publication,loaded};
}


std::vector<int64_t> launch_resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,generate_kernel<true,GSPIM_GENERATION_TRACE>));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("generate",&generate);m.def("launch_resources",&launch_resources);}
