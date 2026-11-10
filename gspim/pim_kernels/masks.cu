
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <math_constants.h>
using torch::Tensor;
struct Support {float4 u,v,q;};
__global__ void make_support(const float2* xy,const float3* conic,const float* opacity,
    const int2* radii,Support* output,unsigned* boxes,int N,int W,int ts) {
  int g=blockIdx.x*blockDim.x+threadIdx.x;if(g>=N)return;
  int reference=-1;
  for(int f=0;f<W;++f){int2 r=radii[f*N+g];if(r.x>0&&r.y>0){reference=f*N+g;break;}}
  float3 ref=reference>=0?conic[reference]:make_float3(1,0,1);
  bool swap=ref.z>ref.x;
  float ratio=(swap?ref.z:ref.x)>0?__fdividef(ref.y,swap?ref.z:ref.x):0;
  float2 refp=reference>=0?xy[reference]:make_float2(0,0);
  float v0=swap?refp.x:refp.y,extent=0;
  float umin=CUDART_INF_F,umax=-CUDART_INF_F,vmin=CUDART_INF_F,vmax=-CUDART_INF_F;
  float pivot=CUDART_INF_F,other=CUDART_INF_F,cut=-CUDART_INF_F,error=0,dr=0;
  bool valid=true,any=false;
  for(int f=0;f<W;++f) {
    int j=f*N+g;float2 p=xy[j];int2 r=radii[j];float3 a=conic[j];
    unsigned box=0;
    if(r.x>0&&r.y>0) {
      int xmin=max(0,min(255,int(floorf((p.x-r.x)/ts))));
      int xmax=max(0,min(255,int(ceilf((p.x+r.x)/ts))));
      int ymin=max(0,min(255,int(floorf((p.y-r.y)/ts))));
      int ymax=max(0,min(255,int(ceilf((p.y+r.y)/ts))));
      box=unsigned(xmin)|(unsigned(xmax)<<8)|(unsigned(ymin)<<16)|(unsigned(ymax)<<24);
      float pp=swap?a.z:a.x,rr=__fdividef(a.y,pp),oo=(swap?a.x:a.z)-a.y*rr;
      valid&=pp>0&&oo>=0&&isfinite(rr);any=true;
      float pu=swap?p.y:p.x,pv=swap?p.x:p.y;
      float cu=pu+rr*(pv-v0);
      extent=fmaxf(extent,fabsf(pv-v0)+(swap?r.x:r.y)+ts);
      umin=fminf(umin,cu);umax=fmaxf(umax,cu);
      vmin=fminf(vmin,pv);vmax=fmaxf(vmax,pv);
      dr=fmaxf(dr,fabsf(rr-ratio));pivot=fminf(pivot,pp);other=fminf(other,oo);
      cut=fmaxf(cut,__logf(255.f*opacity[j])+1e-4f);
      float mx=float(r.x)+ts,my=float(r.y)+ts;
      error=fmaxf(error,2e-6f*(fabsf(a.x)*mx*mx+fabsf(a.z)*my*my+2*fabsf(a.y)*mx*my));
    }
    boxes[f*N+g]=box;
  }
  // A selected tile lies within a visible radius plus one tile of v0.
  float margin=dr*extent+2e-3f*(1+fabsf(ratio));
  output[g]={make_float4(umin-margin,umax+margin,ratio,pivot),
             make_float4(vmin-.002f,vmax+.002f,other,cut),
             make_float4(error+1e-3f,v0,0,__uint_as_float(unsigned(swap)|(unsigned(valid&&any)<<1)))};
}
__global__ void flags_kernel(const int* keys,int* ids,int* flags,const Support* support,
    const unsigned* boxes,int n,int N,int W,int bits,int tw,int shift,int drop,
    unsigned long long* publication,const int2* physical,int physical_slots,
    const unsigned long long* source_publication,const unsigned long long* delay,
    int2* physical_mask,unsigned long long* reads,const int4* directory,int* grouped_ids,bool physical_only,bool compact,const int* original_indices,int* smids) {
  int i=blockIdx.x*256+threadIdx.x;
  int b=blockIdx.x,owner=(b%32)*12+(b/32)%12,slot=(b/384)*256+threadIdx.x;
  int source_block=b;bool live=i<n;
  if(directory){int4 d=directory[b];owner=d.x;slot=d.y+threadIdx.x;source_block=d.z;live=threadIdx.x<d.w;}
  if(physical){
    if(!threadIdx.x){unsigned long long now,target=source_publication[source_block*2+1]+delay[b];
      do{asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));if(now<target)__nanosleep(64);}while(now<target);reads[b]=now;}
    __syncthreads();
  }
  if(publication&&!threadIdx.x) {
    unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));smids[blockIdx.x]=sm;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2]));
  }
  if(live) {
    int2 pair;
    int sorted=0;
    if(directory){if(compact){pair=physical[owner*physical_slots+slot];if(!physical_only)i=original_indices[owner*physical_slots+slot];}
      else{int4 value=((const int4*)physical)[owner*physical_slots+slot];pair=make_int2(value.x,value.y);i=value.z;sorted=value.w;}}
    else pair=physical?physical[owner*physical_slots+slot]:make_int2(keys[i],ids[i]);
    int tile=(pair.x>>drop)&((1<<shift)-1),x=tile%tw,y=tile/tw,g=pair.y;
    unsigned mask=0,regions=0;
    for(int f=0;f<W;++f) {
      unsigned b=boxes[f*N+g];
      if(x>=(b&255u)&&x<((b>>8)&255u)&&y>=((b>>16)&255u)&&y<(b>>24))mask|=1u<<f;
    }
    if(mask) {
      Support s=support[g];bool swap=__float_as_uint(s.q.w)&1u,valid=__float_as_uint(s.q.w)&2u;
      #pragma unroll
      for(int part=0;part<8;++part) {
        float px=x*16+(part%2)*8+4.f,py=y*16+(part/2)*4+2.f;
        float zu=swap?py:px,zv=swap?px:py,ru=swap?1.5f:3.5f,rv=swap?3.5f:1.5f;
        float center=zu+s.u.z*(zv-s.q.y),radius=ru+fabsf(s.u.z)*rv;
        float d=fmaxf(0,fmaxf(center-radius-s.u.y,s.u.x-center-radius));
        float e=fmaxf(0,fmaxf(zv-rv-s.v.y,s.v.x-zv-rv));
        float lower=.5f*(s.u.w*d*d+s.v.z*e*e)-s.q.x;
        if(!valid||!(lower>s.v.w))regions|=1u<<part;
      }
    }
    int flag=mask&&regions,encoded=g|(mask<<bits)|(regions<<(bits+W));
    if(grouped_ids&&!flag)encoded=0;
    if(!physical_only){flags[i]=flag;ids[i]=encoded;}
    if(grouped_ids){
      int block=sorted/256,bank_owner=(block%32)*12+(block/32)%12;
      int bank_slots=((n+98303)/98304)*256,bank_slot=(block/384)*256+sorted%256;
      asm volatile("st.global.wt.u32 [%0], %1;"::"l"(grouped_ids+bank_owner*bank_slots+bank_slot),"r"(encoded):"memory");
    }
    if(physical_mask){int* p=compact?(int*)physical_mask+owner*physical_slots+slot:(int*)(physical_mask+owner*physical_slots+slot);
      if(compact){encoded=flag?encoded:0;asm volatile("st.global.wt.u32 [%0], %1;"::"l"(p),"r"(encoded):"memory");}
      else{
      asm volatile("st.global.wt.u32 [%0], %1;"::"l"(p),"r"(flag):"memory");
      asm volatile("st.global.wt.u32 [%0], %1;"::"l"(p+1),"r"(encoded):"memory");}}
  }
  if(publication) {
    __syncthreads();
    if(!threadIdx.x){__threadfence_system();asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2+1]));}
  }
}
std::vector<Tensor> cull_masks(Tensor keys,Tensor ids,Tensor xy,Tensor conic,Tensor opacity,
    int tw,int th,int ts,int drop,Tensor radii,int bits,bool publication,std::vector<Tensor> supply,bool physical_only) {
  c10::cuda::CUDAGuard guard(xy.device());int N=xy.size(1),W=xy.size(0),n=ids.numel(),shift=0;
  TORCH_CHECK(keys.scalar_type()==torch::kInt32&&ts==16&&tw<256&&th<256&&bits+W+8<=32,"shared int32 tile inputs required");
  while((1<<shift)<=tw*th)++shift;
  auto data=torch::empty({N,12},xy.options());auto boxes=torch::empty({W,N},ids.options());
  int blocks=supply.size()>=6?supply[5].size(0):(n+255)/256;
  auto flags=torch::empty_like(ids),timer=torch::empty({publication?blocks:0,2},ids.options().dtype(torch::kInt64));
  auto smids=torch::empty({publication?blocks:0},ids.options());
  auto stream=at::cuda::getCurrentCUDAStream();
  if(N)make_support<<<(N+255)/256,256,0,stream>>>((float2*)xy.data_ptr<float>(),(float3*)conic.data_ptr<float>(),opacity.data_ptr<float>(),(int2*)radii.data_ptr<int>(),(Support*)data.data_ptr<float>(),(unsigned*)boxes.data_ptr<int>(),N,W,ts);
  if(n)flags_kernel<<<blocks,256,0,stream>>>(keys.data_ptr<int>(),ids.data_ptr<int>(),flags.data_ptr<int>(),(Support*)data.data_ptr<float>(),(unsigned*)boxes.data_ptr<int>(),n,N,W,bits,tw,shift,drop,publication?(unsigned long long*)timer.data_ptr<int64_t>():nullptr,
    supply.empty()?nullptr:(int2*)supply[0].data_ptr<int>(),supply.empty()?0:supply[0].size(1),
    supply.empty()?nullptr:(unsigned long long*)supply[1].data_ptr<int64_t>(),
    supply.empty()?nullptr:(unsigned long long*)supply[2].data_ptr<int64_t>(),
    supply.empty()?nullptr:(int2*)supply[3].data_ptr<int>(),
    supply.empty()?nullptr:(unsigned long long*)supply[4].data_ptr<int64_t>(),
    supply.size()>=6?(int4*)supply[5].data_ptr<int>():nullptr,
    supply.size()>=7&&supply[6].numel()?supply[6].data_ptr<int>():nullptr,physical_only,!supply.empty()&&supply[3].size(2)==1,
    supply.size()==8?supply[7].data_ptr<int>():nullptr,publication?smids.data_ptr<int>():nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(publication)return {flags,ids,timer,smids};
  return {flags,ids,timer};
}
std::vector<int64_t> launch_resources() {
  cudaFuncAttributes attributes;cudaDeviceProp device;int id;
  C10_CUDA_CHECK(cudaGetDevice(&id));C10_CUDA_CHECK(cudaGetDeviceProperties(&device,id));
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,flags_kernel));
  return {attributes.numRegs,int64_t(attributes.sharedSizeBytes),int64_t(attributes.localSizeBytes),
    attributes.maxThreadsPerBlock,device.maxThreadsPerMultiProcessor,device.regsPerMultiprocessor,
    int64_t(device.sharedMemPerMultiprocessor),device.maxBlocksPerMultiProcessor,device.warpSize,device.multiProcessorCount};
}
template <typename Key>
__global__ void compact_tiles(const Key *keys, const int *ids,
                              const int *flags, const int *scan, int n,
                              Key *output_keys, int *output_ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && flags[i]) {
    output_keys[scan[i]] = keys[i];
    output_ids[scan[i]] = ids[i];
  }
}

std::vector<Tensor> compact(Tensor keys,Tensor ids,Tensor flags) {
  c10::cuda::CUDAGuard guard(keys.device());int n=ids.numel();
  if(!n)return {keys,ids};
  auto scan=torch::empty_like(flags);auto stream=at::cuda::getCurrentCUDAStream();
  size_t bytes=0;
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,bytes,flags.data_ptr<int>(),scan.data_ptr<int>(),n,stream));
  auto tmp=torch::empty({int64_t(bytes)},ids.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(tmp.data_ptr(),bytes,flags.data_ptr<int>(),scan.data_ptr<int>(),n,stream));
  int last[2];
  C10_CUDA_CHECK(cudaMemcpyAsync(last,scan.data_ptr<int>()+n-1,4,cudaMemcpyDeviceToHost,stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(last+1,flags.data_ptr<int>()+n-1,4,cudaMemcpyDeviceToHost,stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  auto out_keys=torch::empty({last[0]+last[1]},keys.options()),out_ids=torch::empty({last[0]+last[1]},ids.options());
  compact_tiles<int><<<(n+255)/256,256,0,stream>>>(keys.data_ptr<int>(),ids.data_ptr<int>(),flags.data_ptr<int>(),scan.data_ptr<int>(),n,out_keys.data_ptr<int>(),out_ids.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {out_keys,out_ids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("cull_masks",&cull_masks);m.def("compact",&compact);m.def("launch_resources",&launch_resources);}
