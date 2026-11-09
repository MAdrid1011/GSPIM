
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <math_constants.h>
using torch::Tensor;
__device__ unsigned long long timestamp() {
  unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(value));return value;
}

__device__ void store_wt(unsigned* address,unsigned value) {
  asm volatile("st.global.wt.u32 [%0], %1;"::"l"(address),"r"(value):"memory");
}
template <bool Streaming = false, bool IndexedSource = false, bool Compact = false,
          bool Temporal = false>
__global__ void batch_colors_kernel(const float3 *means, const int2 *radii,
                                    const float *sh, const float *camera,
                                    const int64_t *order, int start, int count,
                                    int N, int W, float *colors,
                                    const float2 *xy, const float3 *conics,
                                    const float *opacity, float *packed,
                                    int stride, const int64_t *source_ids,
                                    int source_count, const int *layout,
                                    const float *frames, const float *temporal_mu,unsigned* physical,
float* logarithm,unsigned* visibility,int slots,unsigned long long* publication,
unsigned long long* load_frontier) {
  __shared__ unsigned long long sh_clock[192];
  sh_clock[threadIdx.x]=0;
  if(!threadIdx.x) {
    publication[blockIdx.x*3]=timestamp();unsigned sm;
    asm volatile("mov.u32 %0, %%smid;":"=r"(sm));publication[blockIdx.x*3+2]=sm;
  }
  do {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * 3)
    break;
  int rank = start + i / 3, channel = i % 3;
  int g = order ? order[rank] : rank;
  if (radii) {
    bool visible = false;
    for (int f = 0; f < W; ++f)
      visible |= radii[f * N + g].x > 0 && radii[f * N + g].y > 0;
    if (!visible) {
      for (int f = 0; f < W; ++f)
        colors[3 * (f * N + g) + channel] = .5f;
      break;
    }
  }
  constexpr int Coefficients = Temporal ? 48 : 16;
  float coefficients[Coefficients];
  int source = IndexedSource ? source_ids[g] : g;
#pragma unroll
  for (int k = 0; k < Coefficients; ++k)
    coefficients[k] = Streaming ? __ldcs(sh + source * Coefficients * 3 + k * 3 + channel)
                                : sh[source * Coefficients * 3 + k * 3 + channel];
  asm volatile("" :: "f"(coefficients[0]),"f"(coefficients[1]),"f"(coefficients[2]),"f"(coefficients[3]),"f"(coefficients[4]),"f"(coefficients[5]),"f"(coefficients[6]),"f"(coefficients[7]),"f"(coefficients[8]),"f"(coefficients[9]),"f"(coefficients[10]),"f"(coefficients[11]),"f"(coefficients[12]),"f"(coefficients[13]),"f"(coefficients[14]),"f"(coefficients[15]),"f"(coefficients[16]),"f"(coefficients[17]),"f"(coefficients[18]),"f"(coefficients[19]),"f"(coefficients[20]),"f"(coefficients[21]),"f"(coefficients[22]),"f"(coefficients[23]),"f"(coefficients[24]),"f"(coefficients[25]),"f"(coefficients[26]),"f"(coefficients[27]),"f"(coefficients[28]),"f"(coefficients[29]),"f"(coefficients[30]),"f"(coefficients[31]),"f"(coefficients[32]),"f"(coefficients[33]),"f"(coefficients[34]),"f"(coefficients[35]),"f"(coefficients[36]),"f"(coefficients[37]),"f"(coefficients[38]),"f"(coefficients[39]),"f"(coefficients[40]),"f"(coefficients[41]),"f"(coefficients[42]),"f"(coefficients[43]),"f"(coefficients[44]),"f"(coefficients[45]),"f"(coefficients[46]),"f"(coefficients[47]) : "memory");
  sh_clock[threadIdx.x]=timestamp();
  for (int f = 0; f < W; ++f) {
    int j = f * N + g;
    if (radii && (radii[j].x <= 0 || radii[j].y <= 0)) {
      colors[3 * j + channel] = .5f;
      continue;
    }
    int mean_id = Temporal ? source : (IndexedSource ? f * source_count + source : j);
    float3 p = Streaming ? make_float3(__ldcs(&means[mean_id].x), __ldcs(&means[mean_id].y), __ldcs(&means[mean_id].z)) : means[mean_id];
    float x = p.x - camera[0], y = p.y - camera[1], z = p.z - camera[2];
    float inv = rsqrtf(x * x + y * y + z * z);
    x *= inv;
    y *= inv;
    z *= inv;
    float xx = x * x, yy = y * y, zz = z * z;
    float basis[16] = {.28209479177387814f,
                       -.4886025119029199f * y,
                       .4886025119029199f * z,
                       -.4886025119029199f * x,
                       1.0925484305920792f * x * y,
                       -1.0925484305920792f * y * z,
                       .31539156525252005f * (2 * zz - xx - yy),
                       -1.0925484305920792f * x * z,
                       .5462742152960396f * (xx - yy),
                       -.5900435899266435f * y * (3 * xx - yy),
                       2.890611442640554f * x * y * z,
                       -.4570457994644658f * y * (4 * zz - xx - yy),
                       .3731763325901154f * z * (2 * zz - 3 * xx - 3 * yy),
                       -.4570457994644658f * x * (4 * zz - xx - yy),
                       1.445305721320277f * z * (xx - yy),
                       -.5900435899266435f * x * (xx - 3 * yy)};
    float value = 0;
#pragma unroll
    for (int k = 0; k < 16; ++k)
      value += basis[k] * coefficients[k];
    if (Temporal) {
      float first = 0, second = 0;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        first += basis[k]*coefficients[16+k];
        second += basis[k]*coefficients[32+k];
      }
      float phase = 2*CUDART_PI_F*(temporal_mu[source]-frames[f]/30.0f)/10.0f;
      value += cosf(phase)*first;
      value += cosf(2*phase)*second;
    }

      colors[3 * j + channel] = fmaxf(0, value + .5f);
  }

  } while(false);
  __syncthreads();
  if(threadIdx.x<64) {
    int g=blockIdx.x*64+threadIdx.x;
    int block=g/256,owner=block%384,slot=(block/384)*256+g%256;
    for(int f=0;f<W;++f) {
      bool valid=g<N&&radii[f*N+g].x>0&&radii[f*N+g].y>0;
      unsigned mask=__ballot_sync(0xffffffff,valid);
      if(!(threadIdx.x%32))store_wt(visibility+(f*((N+255)/256)+block)*8+(g%256)/32,mask);
      if(blockIdx.x+1==gridDim.x&&!threadIdx.x)
        for(int word=(g%256)/32+2;word<8;++word)
          store_wt(visibility+(f*((N+255)/256)+block)*8+word,0);
      if(g<N) {
        int j=f*N+g;unsigned* base=physical+(owner*W+f)*slots*10;
        float2 p=xy[j];float3 a=conics[j],c=((const float3*)colors)[j];
        float op=opacity[j],cutoff=__log2f(op);logarithm[j]=cutoff;
        store_wt(base+slot*2,__float_as_uint(p.x));store_wt(base+slot*2+1,__float_as_uint(p.y));
        store_wt(base+slots*2+slot*3,__float_as_uint(a.x));
        store_wt(base+slots*2+slot*3+1,__float_as_uint(a.y));
        store_wt(base+slots*2+slot*3+2,__float_as_uint(a.z));
        store_wt(base+slots*5+slot*3,__float_as_uint(c.x));
        store_wt(base+slots*5+slot*3+1,__float_as_uint(c.y));
        store_wt(base+slots*5+slot*3+2,__float_as_uint(c.z));
        store_wt(base+slots*8+slot,__float_as_uint(op));
        store_wt(base+slots*9+slot,__float_as_uint(cutoff));
      }
    }
    __threadfence_system();
  }
  __syncthreads();
  if(!threadIdx.x) {
    unsigned long long latest=publication[blockIdx.x*3];
    for(int t=0;t<192;++t)latest=max(latest,sh_clock[t]);
    load_frontier[blockIdx.x]=latest;
    publication[blockIdx.x*3+1]=timestamp();
  }
}

std::vector<Tensor> run(Tensor means,Tensor radii,Tensor sh,Tensor camera,
    Tensor xy,Tensor conics,Tensor opacity,Tensor frames,Tensor temporal_mu) {
  c10::cuda::CUDAGuard guard(means.device());int N=means.size(1),W=opacity.size(0);
  int slots=((N+98303)/98304)*256;
  TORCH_CHECK(means.is_contiguous()&&radii.is_contiguous()&&xy.is_contiguous()&&
    conics.is_contiguous()&&opacity.is_contiguous(),"contiguous window required");
  auto colors=torch::empty({W,N,3},means.options()),logarithm=torch::empty_like(opacity);
  auto physical=torch::empty({384,W,slots*10},radii.options());
  auto visibility=torch::empty({W,(N+255)/256,8},radii.options());
  auto publication=torch::empty({(N+63)/64,3},means.options().dtype(torch::kInt64));
  auto load_frontier=torch::empty({(N+63)/64},means.options().dtype(torch::kInt64));
  if(N)batch_colors_kernel<false,false,false,true><<<(N+63)/64,192,0,at::cuda::getCurrentCUDAStream()>>>(
    (float3*)means.data_ptr<float>(),(int2*)radii.data_ptr<int>(),sh.data_ptr<float>(),
    camera.data_ptr<float>(),nullptr,0,N,N,W,colors.data_ptr<float>(),
    (float2*)xy.data_ptr<float>(),(float3*)conics.data_ptr<float>(),opacity.data_ptr<float>(),
    nullptr,0,nullptr,0,nullptr,frames.data_ptr<float>(),temporal_mu.data_ptr<float>(),(unsigned*)physical.data_ptr<int>(),
    logarithm.data_ptr<float>(),(unsigned*)visibility.data_ptr<int>(),slots,
    (unsigned long long*)publication.data_ptr<int64_t>(),
    (unsigned long long*)load_frontier.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {colors,logarithm,physical,visibility,publication,load_frontier};
}

std::vector<int64_t> resources() {
  cudaFuncAttributes a;cudaDeviceProp d;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&d,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,batch_colors_kernel<false,false,false,true>));
  return {a.numRegs,int64_t(a.sharedSizeBytes),int64_t(a.localSizeBytes),a.maxThreadsPerBlock,
    d.maxThreadsPerMultiProcessor,d.regsPerMultiprocessor,int64_t(d.sharedMemPerMultiprocessor),
    d.maxBlocksPerMultiProcessor,d.warpSize,d.multiProcessorCount};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);m.def("resources",&resources);}
