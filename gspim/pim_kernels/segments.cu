
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
using torch::Tensor;
struct Candidate { float4 geometry, color, extra; };
__device__ __forceinline__ float supported_exp(float exponent) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}
__global__ void cutoff_kernel(const float* op,float* cutoff,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)cutoff[i]=__log2f(op[i]);
}
Tensor cutoffs(Tensor opacity) {
  auto out=torch::empty_like(opacity);
  if(opacity.numel())cutoff_kernel<<<(opacity.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    opacity.data_ptr<float>(),out.data_ptr<float>(),opacity.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
__global__ void dense_ranges(const int* offsets,const int64_t* jobs,int* ranges,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int tile=jobs[i];ranges[2*tile]=offsets[i];ranges[2*tile+1]=offsets[i+1];
}
template<int Warps>
__global__ void packed_consume(const Candidate* fields,
    const int* ids,const int* offsets,const int* locations,const int* counts,int N,int bits,int W,int width,int height,float4* image,float background,unsigned long long* trace,unsigned long long* task_trace,
    const unsigned long long* delivery,const unsigned long long* id_ready,const unsigned long long* field_ready) {
  constexpr int Pixels=4;constexpr unsigned AllPixels=85;
  int lane=threadIdx.x%32,local_warp=threadIdx.x/32;
  int task=blockIdx.x*Warps+local_warp;
  int tw=(width+15)/16,tiles=tw*((height+15)/16);
  if(task>=tiles*W*2)return;
  int tile=task/(W*2),f=(task/2)%W,half=task%2;
  int x=(tile%tw)*16+half*8+lane%8,y=(tile/tw)*16+lane/8;

  if(task_trace&&!lane) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));task_trace[task*3+2]=sm;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(task_trace[task*3])::"memory");
  }
  float T[Pixels];float3 C[Pixels];unsigned done_bits=0,dirty_bits=0;
#pragma unroll
  for(int q=0;q<Pixels;++q) {
    if(x>=width||y+q*4>=height)done_bits|=1u<<(q*2);
    T[q]=1;C[q]=make_float3(0,0,0);
  }
  __shared__ Candidate staged[Warps*32];
  float stop_threshold=.0001f;

  for(int segment=offsets[task];segment>=0;segment=offsets[tiles*W*2+segment]) {
    if(__all_sync(0xffffffff,done_bits==AllPixels))break;
    if(trace&&!lane)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(trace[segment*13+12])::"memory");
    if(delivery) {
      if(!lane) {
        unsigned long long due=max(id_ready[delivery[segment*3+2]]+delivery[segment*3],field_ready[0]+delivery[segment*3+1]);
        unsigned long long now;
        do {asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now)::"memory");}while(now<due);
      }
      __syncwarp();
    }
    if(trace&&!lane)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(trace[segment*13])::"memory");

    int location=locations[segment],hi=location*256+counts[location*8],consumed=0;
    for(int base=location*256;base<hi;base+=32) {
    if(__all_sync(0xffffffff,done_bits==AllPixels))break;
    if(trace&&!lane)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(trace[segment*13+4+(base-location*256)/32])::"memory");
    if(base+lane<hi) {
      unsigned encoded=ids[base+lane];int source=encoded&((1u<<bits)-1);
      int g=f*N+source;
      Candidate value=fields[g];
      unsigned mask=(encoded>>(bits+W+half))&0x55u;
      value.extra.z=__uint_as_float(value.color.y>0?mask:0);
      value.geometry.z=__fmul_rn(-.7213475204444817f,value.geometry.z);
      value.geometry.w=__fmul_rn(-1.4426950408889634f,value.geometry.w);
      value.color.x=__fmul_rn(-.7213475204444817f,value.color.x);
      staged[local_warp*32+lane]=value;
    }
    __syncwarp();
    int total=min(32,hi-base);consumed=base-location*256+total;
    #pragma unroll 2
    for(int k=0;k<total;++k) {
      Candidate value=staged[local_warp*32+k];
      unsigned active=__float_as_uint(value.extra.z)&~done_bits;
      float2 p=make_float2(value.geometry.x,value.geometry.y);
      float3 a=make_float3(value.geometry.z,value.geometry.w,value.color.x);
      float3 col=make_float3(value.color.z,value.color.w,value.extra.x);
      float op=value.color.y,cutoff=value.extra.y;
        float dx = p.x - (x + .5f);
        float xhalf = __fmaf_rn(__fmul_rn(a.x,dx),dx,cutoff);
        float yhalf = a.z;
        float cross = __fmul_rn(a.y, dx);
#pragma unroll
        for (int q = 0; q < Pixels; ++q) {
          if (!(active & (1u << (q * 2))))
            continue;
          float dy = p.y - (y + q * 4 + .5f);
          // Share the x contribution across the thread's two pixel rows.
          float sigma = __fmaf_rn(dy, __fmaf_rn(dy, yhalf, cross), xhalf);
          float alpha = fminf(.999f, supported_exp(sigma));
          if (sigma > cutoff || alpha < 1.f / 255.f)
            continue;
          float next = __fmaf_rn(-alpha,T[q],T[q]);
          dirty_bits |= 1u << (q * 2);
          if (stop_threshold > 0 && next <= stop_threshold) {
            done_bits |= 1u << (q * 2);
            continue;
          }
          float w = alpha * T[q];
          C[q].x += col.x * w;
          C[q].y += col.y * w;
          C[q].z += col.z * w;
          T[q] = next;
      }
    }
    __syncwarp();
    }
    if(trace&&!lane) {
      trace[segment*13+2]=consumed;
      unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));trace[segment*13+3]=sm;
      asm volatile("mov.u64 %0, %%globaltimer;":"=l"(trace[segment*13+1])::"memory");
    }
  }
#pragma unroll
  for(int q=0;q<Pixels;++q)if(x<width&&y+q*4<height) {
    int pixel=(f*height+y+q*4)*width+x;float b=background*T[q];
    __stcs(image+pixel,make_float4(C[q].x+b,C[q].y+b,C[q].z+b,0));
  }
  if(task_trace&&!lane)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(task_trace[task*3+1])::"memory");
}
__global__ void field_kernel(const float2* xy,const float3* conic,
    const float3* colors,const float* opacity,const float* cutoff,Candidate* output,int n,int N,int capacity) {
  int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=n)return;
  float2 p=xy[g];float3 a=conic[g],c=colors[g];
  int target=g;
  output[target]={make_float4(p.x,p.y,a.x,a.y),make_float4(a.z,opacity[g],c.x,c.y),
             make_float4(c.z,cutoff[g],0,0)};
}
Tensor fields(Tensor xy,Tensor conic,Tensor colors,Tensor opacity,Tensor cutoff) {
  c10::cuda::CUDAGuard guard(xy.device());int n=opacity.numel();
  int N=xy.size(1),capacity=N;
  auto output=torch::empty({n,12},xy.options());
  if(n)field_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float2*)xy.data_ptr<float>(),(float3*)conic.data_ptr<float>(),(float3*)colors.data_ptr<float>(),
    opacity.data_ptr<float>(),cutoff.data_ptr<float>(),(Candidate*)output.data_ptr<float>(),n,N,capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
__device__ void store_wt(unsigned* address,unsigned value) {
  asm volatile("st.global.wt.u32 [%0], %1;"::"l"(address),"r"(value):"memory");
}
__global__ void source_fields(const float2* xy,const float3* conic,const float3* rgb,
    const float* opacity,const float* logarithm,unsigned* output,int N,int W,int slots) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*W)return;
  int g=i%N,f=i/N,block=g/256,owner=block%384,slot=(block/384)*256+g%256;
  unsigned* base=output+(owner*W+f)*slots*10;
  float2 p=xy[i];float3 a=conic[i],c=rgb[i];
  store_wt(base+slot*2,__float_as_uint(p.x));store_wt(base+slot*2+1,__float_as_uint(p.y));
  store_wt(base+slots*2+slot*3,__float_as_uint(a.x));store_wt(base+slots*2+slot*3+1,__float_as_uint(a.y));
  store_wt(base+slots*2+slot*3+2,__float_as_uint(a.z));
  store_wt(base+slots*5+slot*3,__float_as_uint(c.x));store_wt(base+slots*5+slot*3+1,__float_as_uint(c.y));
  store_wt(base+slots*5+slot*3+2,__float_as_uint(c.z));
  store_wt(base+slots*8+slot,__float_as_uint(opacity[i]));
  store_wt(base+slots*9+slot,__float_as_uint(logarithm[i]));
}
Tensor emit_fields(Tensor xy,Tensor conic,Tensor rgb,Tensor opacity,Tensor logarithm) {
  c10::cuda::CUDAGuard guard(xy.device());int N=xy.size(1),W=xy.size(0),slots=((N+98303)/98304)*256;
  auto output=torch::empty({384,W,slots*10},xy.options().dtype(torch::kInt32));
  if(N*W)source_fields<<<(N*W+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float2*)xy.data_ptr<float>(),(float3*)conic.data_ptr<float>(),(float3*)rgb.data_ptr<float>(),
    opacity.data_ptr<float>(),logarithm.data_ptr<float>(),(unsigned*)output.data_ptr<int>(),N,W,slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
__global__ void segment_counts(const int* dense,int* counts,int tasks,int W) {
  int task=blockIdx.x*blockDim.x+threadIdx.x;if(task>=tasks)return;
  int tile=task/(W*2),lo=dense[tile*2],hi=dense[tile*2+1];
  counts[task]=hi>lo?(hi-1)/256-lo/256+1:0;
}
__global__ void matrix_counts(const int* dense,const int* starts,int* counts,int tasks,int W) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tasks*32)return;
  int task=i%tasks,die=i/tasks,tile=task/(W*2),first=dense[tile*2]/256;
  int n=starts[task+1]-starts[task],distance=(die-(first%32)+32)%32;
  counts[i]=n/32+(distance<n%32);
}
__global__ void segment_descriptors(const int* dense,const int* starts,const int* positions,
    int4* descriptors,int* locations,int4* routed,int tasks,int W) {
  int task=blockIdx.x*blockDim.x+threadIdx.x;if(task>=tasks)return;
  int tile=task/(W*2),lo=dense[tile*2],hi=dense[tile*2+1],first=lo/256;
  for(int s=starts[task];s<starts[task+1];++s) {
    int block=first+s-starts[task],die=block%32;
    int location=positions[die*tasks+task]+(block-first)/32;
    int a=max(lo,block*256),b=min(hi,(block+1)*256);
    descriptors[s]=make_int4(a,b,task,die);locations[s]=location;
    routed[location]=make_int4(a,b,task,s);
  }
}
__global__ void die_boundaries(const int* positions,int* starts,int tasks) {
  int d=threadIdx.x;if(d<=32)starts[d]=positions[d*tasks];
}
__global__ void scatter_ids(const int* input,unsigned* output,int n,int slots) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  int block=i/256,owner=(block%32)*12+(block/32)%12,slot=(block/384)*256+i%256;
  store_wt(output+owner*slots+slot,unsigned(input[i]));
}
Tensor emit_ids(Tensor input) {
  c10::cuda::CUDAGuard guard(input.device());int n=input.numel(),slots=((n+98303)/98304)*256;
  auto output=torch::empty({384,slots},input.options());
  if(n)scatter_ids<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    input.data_ptr<int>(),(unsigned*)output.data_ptr<int>(),n,slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return output;
}
Tensor exclusive(Tensor counts) {
  auto result=torch::empty_like(counts);size_t bytes=0;auto stream=at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,bytes,counts.data_ptr<int>(),result.data_ptr<int>(),counts.numel(),stream));
  auto scratch=torch::empty({int64_t(bytes)},counts.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(scratch.data_ptr(),bytes,counts.data_ptr<int>(),result.data_ptr<int>(),counts.numel(),stream));
  return result;
}
std::vector<Tensor> plan(Tensor offsets,Tensor jobs,Tensor input,int W,int width,int height) {
  c10::cuda::CUDAGuard guard(offsets.device());
  int tiles=((width+15)/16)*((height+15)/16),tasks=tiles*W*2,input_count=input.numel();
  int capacity=((input_count+255)/256+tiles)*W*2,slots=((input_count+98303)/98304)*256;
  auto dense=torch::zeros({tiles*2},offsets.options());
  auto counts=torch::zeros({tasks+1},offsets.options());
  auto matrix=torch::zeros({tasks*32+1},offsets.options());
  auto descriptors=torch::empty({capacity,4},offsets.options()),routed=torch::empty_like(descriptors);
  auto locations=torch::empty({capacity},offsets.options()),die_starts=torch::empty({33},offsets.options());
  auto physical_ids=torch::empty({384,slots},offsets.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  if(jobs.numel())dense_ranges<<<(jobs.numel()+255)/256,256,0,stream>>>(offsets.data_ptr<int>(),jobs.data_ptr<int64_t>(),dense.data_ptr<int>(),jobs.numel());
  segment_counts<<<(tasks+255)/256,256,0,stream>>>(dense.data_ptr<int>(),counts.data_ptr<int>(),tasks,W);
  auto starts=exclusive(counts);
  matrix_counts<<<(tasks*32+255)/256,256,0,stream>>>(dense.data_ptr<int>(),starts.data_ptr<int>(),matrix.data_ptr<int>(),tasks,W);
  auto positions=exclusive(matrix);
  segment_descriptors<<<(tasks+255)/256,256,0,stream>>>(dense.data_ptr<int>(),starts.data_ptr<int>(),positions.data_ptr<int>(),
    (int4*)descriptors.data_ptr<int>(),locations.data_ptr<int>(),(int4*)routed.data_ptr<int>(),tasks,W);
  die_boundaries<<<1,64,0,stream>>>(positions.data_ptr<int>(),die_starts.data_ptr<int>(),tasks);
  if(input_count)scatter_ids<<<(input_count+255)/256,256,0,stream>>>(input.data_ptr<int>(),(unsigned*)physical_ids.data_ptr<int>(),input_count,slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {starts,descriptors,locations,die_starts,routed,physical_ids};
}
__global__ void segment_pack(const int* input,const int4* descriptors,const int* locations,
    int* output,int* counts,int segments,int W,int bits,int slots) {
  int segment=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(segment>=segments)return;
  int4 d=descriptors[segment];int f=(d.z/2)%W,half=d.z%2,cursor=0,location=locations[segment];
  for(int base=d.x;base<d.y;base+=32) {
    int index=base+lane,block=index/256,owner=(block%32)*12+(block/32)%12,slot=(block/384)*256+index%256;
    bool valid=index<d.y;unsigned encoded=valid?unsigned(input[owner*slots+slot]):0;
    bool keep=valid&&((encoded>>bits)&(1u<<f))&&((encoded>>(bits+W+half))&0x55u);
    unsigned mask=__ballot_sync(0xffffffff,keep);
    if(keep)output[location*256+cursor+__popc(mask&((1u<<lane)-1))]=encoded;
    cursor+=__popc(mask);
  }
  if(!lane)counts[location*8]=cursor;
}
std::vector<Tensor> pack_segments(Tensor ids,Tensor descriptors,Tensor locations,int segments,int W,int bits) {
  c10::cuda::CUDAGuard guard(ids.device());
  auto output=torch::empty({segments*256},ids.options()),counts=torch::zeros({segments,8},ids.options());
  if(segments)segment_pack<<<(segments+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),(int4*)descriptors.data_ptr<int>(),
    locations.data_ptr<int>(),output.data_ptr<int>(),counts.data_ptr<int>(),segments,W,bits,ids.numel()/384);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {counts,output};
}
__global__ void publication_kernel(unsigned long long* timer) {
  __threadfence_system();asm volatile("mov.u64 %0, %%globaltimer;":"=l"(timer[0]));
}
Tensor publish(Tensor reference) {
  c10::cuda::CUDAGuard guard(reference.device());
  auto timer=torch::empty({1},reference.options().dtype(torch::kInt64));
  publication_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>((unsigned long long*)timer.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return timer;
}
void consume(Tensor starts,Tensor locations,Tensor counts,Tensor ids,Tensor fields,int W,int width,int height,
    int bits,int N,Tensor image,float background,Tensor trace,Tensor task_trace,Tensor delivery,Tensor id_ready,Tensor field_ready) {
  c10::cuda::CUDAGuard guard(image.device());
  int tasks=((width+15)/16)*((height+15)/16)*W*2;
  TORCH_CHECK(image.dim()==4&&image.size(3)==4&&image.is_contiguous(),"padded output required");
  packed_consume<4><<<(tasks+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(
    (Candidate*)fields.data_ptr<float>(),ids.data_ptr<int>(),starts.data_ptr<int>(),locations.data_ptr<int>(),counts.data_ptr<int>(),
    N,bits,W,width,height,(float4*)image.data_ptr<float>(),background,
    trace.numel()?(unsigned long long*)trace.data_ptr<int64_t>():nullptr,
    task_trace.numel()?(unsigned long long*)task_trace.data_ptr<int64_t>():nullptr,
    delivery.numel()?(unsigned long long*)delivery.data_ptr<int64_t>():nullptr,
    (unsigned long long*)id_ready.data_ptr<int64_t>(),(unsigned long long*)field_ready.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void source_fields_fused(const float2* xy,const float3* conic,const float3* rgb,
    const float* opacity,float* logarithm,unsigned* output,int N,int W,int slots) {
  int block=blockIdx.x,f=blockIdx.y,g=block*256+threadIdx.x;if(g>=N)return;
  int i=f*N+g,owner=block%384,slot=(block/384)*256+threadIdx.x;
  logarithm[i]=__log2f(opacity[i]);
  unsigned* base=output+(owner*W+f)*slots*10;
  float2 p=xy[i];float3 a=conic[i],c=rgb[i];
  store_wt(base+slot*2,__float_as_uint(p.x));store_wt(base+slot*2+1,__float_as_uint(p.y));
  store_wt(base+slots*2+slot*3,__float_as_uint(a.x));store_wt(base+slots*2+slot*3+1,__float_as_uint(a.y));
  store_wt(base+slots*2+slot*3+2,__float_as_uint(a.z));
  store_wt(base+slots*5+slot*3,__float_as_uint(c.x));store_wt(base+slots*5+slot*3+1,__float_as_uint(c.y));
  store_wt(base+slots*5+slot*3+2,__float_as_uint(c.z));
  store_wt(base+slots*8+slot,__float_as_uint(opacity[i]));
  store_wt(base+slots*9+slot,__float_as_uint(logarithm[i]));
}

std::vector<Tensor> emit_fields_fused(Tensor xy,Tensor conic,Tensor rgb,Tensor opacity) {
  c10::cuda::CUDAGuard guard(xy.device());int N=xy.size(1),W=xy.size(0),slots=((N+98303)/98304)*256;
  auto output=torch::empty({384,W,slots*10},xy.options().dtype(torch::kInt32));
  auto logarithm=torch::empty_like(opacity);
  if(N*W)source_fields_fused<<<dim3((N+255)/256,W),256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float2*)xy.data_ptr<float>(),(float3*)conic.data_ptr<float>(),(float3*)rgb.data_ptr<float>(),
    opacity.data_ptr<float>(),logarithm.data_ptr<float>(),(unsigned*)output.data_ptr<int>(),N,W,slots);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {logarithm,output};
}

__global__ void source_fields_selected(const float2* xy,const float3* conic,const float3* rgb,
    const float* opacity,float* logarithm,unsigned* output,int N,int W,int slots,const int2* radii,unsigned* visibility) {

  int block=blockIdx.x,f=blockIdx.y,g=block*256+threadIdx.x;
  bool valid=g<N&&radii[f*N+g].x>0&&radii[f*N+g].y>0;
  unsigned mask=__ballot_sync(0xffffffff,valid);
  if(!(threadIdx.x%32))store_wt(visibility+(f*((N+255)/256)+block)*8+threadIdx.x/32,mask);
  if(g>=N)return;
  int i=f*N+g,owner=block%384,slot=(block/384)*256+threadIdx.x;
  logarithm[i]=__log2f(opacity[i]);
  unsigned* base=output+(owner*W+f)*slots*10;
  float2 p=xy[i];float3 a=conic[i],c=rgb[i];
  store_wt(base+slot*2,__float_as_uint(p.x));store_wt(base+slot*2+1,__float_as_uint(p.y));
  store_wt(base+slots*2+slot*3,__float_as_uint(a.x));store_wt(base+slots*2+slot*3+1,__float_as_uint(a.y));
  store_wt(base+slots*2+slot*3+2,__float_as_uint(a.z));
  store_wt(base+slots*5+slot*3,__float_as_uint(c.x));store_wt(base+slots*5+slot*3+1,__float_as_uint(c.y));
  store_wt(base+slots*5+slot*3+2,__float_as_uint(c.z));
  store_wt(base+slots*8+slot,__float_as_uint(opacity[i]));
  store_wt(base+slots*9+slot,__float_as_uint(logarithm[i]));
}

std::vector<Tensor> emit_fields_selected(Tensor xy,Tensor conic,Tensor rgb,Tensor opacity,Tensor radii) {
  c10::cuda::CUDAGuard guard(xy.device());int N=xy.size(1),W=xy.size(0),slots=((N+98303)/98304)*256;
  auto output=torch::empty({384,W,slots*10},xy.options().dtype(torch::kInt32));
  auto logarithm=torch::empty_like(opacity);
  auto visibility=torch::empty({W,(N+255)/256,8},radii.options());
  if(N*W)source_fields_selected<<<dim3((N+255)/256,W),256,0,at::cuda::getCurrentCUDAStream()>>>(
    (float2*)xy.data_ptr<float>(),(float3*)conic.data_ptr<float>(),(float3*)rgb.data_ptr<float>(),
    opacity.data_ptr<float>(),logarithm.data_ptr<float>(),(unsigned*)output.data_ptr<int>(),N,W,slots,
    (int2*)radii.data_ptr<int>(),(unsigned*)visibility.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {logarithm,output,visibility};
}


__global__ void producer_anchor_kernel(const unsigned long long* publication,const int4* directory,
    unsigned long long* anchors,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)atomicMax(anchors+directory[i].x,publication[i*2+1]);
}
Tensor producer_anchors(Tensor publication,Tensor directory) {
  auto result=torch::zeros({32},publication.options());
  if(directory.size(0))producer_anchor_kernel<<<(directory.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
    (unsigned long long*)publication.data_ptr<int64_t>(),(int4*)directory.data_ptr<int>(),
    (unsigned long long*)result.data_ptr<int64_t>(),directory.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}


std::vector<int64_t> launch_resources() {
  cudaFuncAttributes attributes;cudaDeviceProp device;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&device,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,packed_consume<4>));
  return {attributes.numRegs,int64_t(attributes.sharedSizeBytes),int64_t(attributes.localSizeBytes),
    attributes.maxThreadsPerBlock,device.maxThreadsPerMultiProcessor,device.regsPerMultiprocessor,
    int64_t(device.sharedMemPerMultiprocessor),device.maxBlocksPerMultiProcessor,device.warpSize,device.multiProcessorCount};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("launch_resources",&launch_resources);m.def("producer_anchors",&producer_anchors);m.def("emit_fields_selected",&emit_fields_selected);m.def("emit_fields_fused",&emit_fields_fused);
  m.def("cutoffs",&cutoffs);m.def("fields",&fields);m.def("plan",&plan);
  m.def("emit_fields",&emit_fields);m.def("pack_segments",&pack_segments);m.def("publish",&publish);m.def("consume",&consume);
  m.def("emit_ids",&emit_ids);
}
