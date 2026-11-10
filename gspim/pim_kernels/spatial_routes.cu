
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
using torch::Tensor;
__device__ void wt(int* p,int x){asm volatile("st.global.wt.u32 [%0], %1;"::"l"(p),"r"(x):"memory");}
__device__ void wt4(int* p,int a,int b,int c,int d){asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"::"l"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");}
__global__ void counts_kernel(const int* raw,int* counts,int N){int i=blockIdx.x*256+threadIdx.x;if(i<N)counts[i]=raw[i*8+6];}
__global__ void route_kernel(const int* raw,const int* offsets,int* desc,int* bitmap,
    unsigned long long* publication,int N,int capacity,int words,int rw,int rh,bool local){
  int i=blockIdx.x*256+threadIdx.x,lane=threadIdx.x%32,warp=i/32;unsigned owners=0;
  __shared__ unsigned transposed[256];
  if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2]));
  if(i<N){const int* r=raw+i*8;
    if(r[6])for(int ry=r[4]/rh;ry<=(r[5]-1)/rh;++ry)for(int rx=r[2]/rw;rx<=(r[3]-1)/rw;++rx){
      int die=ry*8+rx,xmin=max(r[2],rx*rw),xmax=min(r[3],(rx+1)*rw);
      int ymin=max(r[4],ry*rh),ymax=min(r[5],(ry+1)*rh),nx=r[3]-r[2];
      int first=(local?0:offsets[i])+(ymin-r[4])*nx+xmin-r[2];int* p=desc+(die*capacity+i)*8;
      wt4(p,r[0],xmin,xmax,ymin);wt4(p+4,ymax,first,nx,i);owners|=1u<<die;
    }
  }
  #pragma unroll
  for(int step=0;step<5;++step){
    int distance=1<<step;unsigned masks[5]={0x55555555u,0x33333333u,0x0f0f0f0fu,0x00ff00ffu,0x0000ffffu};
    unsigned mask=masks[step],other=__shfl_xor_sync(0xffffffff,owners,distance);
    owners=(lane&distance)?((owners&~mask)|((other&~mask)>>distance)):
                           ((owners&mask)|((other&mask)<<distance));
  }
  transposed[lane*8+threadIdx.x/32]=owners;__syncthreads();
  int die=threadIdx.x/8,word=blockIdx.x*8+threadIdx.x%8;
  wt(bitmap+die*words+word,int(transposed[threadIdx.x]));
  __threadfence_system();__syncthreads();if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2+1]));
}
std::vector<Tensor> route_into(Tensor raw,int tw,int th,Tensor desc,Tensor bitmap,Tensor publication,bool local=false){
  c10::cuda::CUDAGuard guard(raw.device());int N=raw.size(0),words=bitmap.size(1);
  auto count=torch::empty({N},raw.options()),offset=torch::empty_like(count);
  auto stream=at::cuda::getCurrentCUDAStream();
  if(N){if(!local){counts_kernel<<<(N+255)/256,256,0,stream>>>(raw.data_ptr<int>(),count.data_ptr<int>(),N);
    size_t bytes=0;C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,bytes,count.data_ptr<int>(),offset.data_ptr<int>(),N,stream));
    auto tmp=torch::empty({int64_t(bytes)},raw.options().dtype(torch::kUInt8));
    C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(tmp.data_ptr(),bytes,count.data_ptr<int>(),offset.data_ptr<int>(),N,stream));
    }
    route_kernel<<<(N+255)/256,256,0,stream>>>(raw.data_ptr<int>(),offset.data_ptr<int>(),desc.data_ptr<int>(),bitmap.data_ptr<int>(),
      (unsigned long long*)publication.data_ptr<int64_t>(),N,desc.size(1),words,(tw+7)/8,(th+3)/4,local);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {desc.narrow(1,0,N),bitmap.narrow(1,0,((N+255)/256)*8),publication.narrow(0,0,(N+255)/256)};
}
std::vector<Tensor> route(Tensor raw,int tw,int th,bool local=false){
  int N=raw.size(0),blocks=(N+255)/256;
  return route_into(raw,tw,th,torch::empty({32,N,8},raw.options()),torch::empty({32,blocks*8},raw.options()),
    torch::empty({blocks,2},raw.options().dtype(torch::kInt64)),local);
}
__global__ void fused_route_kernel(const float2* xy,const int2* radii,const int64_t* order,const int64_t* assignment,int* raw,int* desc,int* bitmap,
    unsigned long long* publication,int N,int capacity,int words,int rw,int rh,int S,int W,int tw,int th,int ts){
  int i=blockIdx.x*256+threadIdx.x,lane=threadIdx.x%32,warp=i/32;unsigned owners=0;
  __shared__ unsigned transposed[256];
  if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2]));
  if(i<N){
    int g=order[i],xmin=tw,xmax=0,ymin=th,ymax=0;
    for(int f=0;f<W;++f){int j=f*S+g;float2 p=xy[j];int2 r=radii[j];if(r.x<=0||r.y<=0)continue;
      xmin=min(xmin,max(0,min(tw,int(floorf((p.x-r.x)/ts)))));
      xmax=max(xmax,max(0,min(tw,int(ceilf((p.x+r.x)/ts)))));
      ymin=min(ymin,max(0,min(th,int(floorf((p.y-r.y)/ts)))));
      ymax=max(ymax,max(0,min(th,int(ceilf((p.y+r.y)/ts)))));
    }
    int values[8]={g,int(assignment[i]),xmin,xmax,ymin,ymax,max(0,xmax-xmin)*max(0,ymax-ymin),0};
    const int* r=values;
    wt4(raw+i*8,r[0],r[1],r[2],r[3]);wt4(raw+i*8+4,r[4],r[5],r[6],r[7]);
    if(r[6])for(int ry=r[4]/rh;ry<=(r[5]-1)/rh;++ry)for(int rx=r[2]/rw;rx<=(r[3]-1)/rw;++rx){
      int die=ry*8+rx,xmin=max(r[2],rx*rw),xmax=min(r[3],(rx+1)*rw);
      int ymin=max(r[4],ry*rh),ymax=min(r[5],(ry+1)*rh),nx=r[3]-r[2];
      int first=0+(ymin-r[4])*nx+xmin-r[2];int* p=desc+(die*capacity+i)*8;
      wt4(p,r[0],xmin,xmax,ymin);wt4(p+4,ymax,first,nx,i);owners|=1u<<die;
    }
  }
  #pragma unroll
  for(int step=0;step<5;++step){
    int distance=1<<step;unsigned masks[5]={0x55555555u,0x33333333u,0x0f0f0f0fu,0x00ff00ffu,0x0000ffffu};
    unsigned mask=masks[step],other=__shfl_xor_sync(0xffffffff,owners,distance);
    owners=(lane&distance)?((owners&~mask)|((other&~mask)>>distance)):
                           ((owners&mask)|((other&mask)<<distance));
  }
  transposed[lane*8+threadIdx.x/32]=owners;__syncthreads();
  int die=threadIdx.x/8,word=blockIdx.x*8+threadIdx.x%8;
  wt(bitmap+die*words+word,int(transposed[threadIdx.x]));
  __threadfence_system();__syncthreads();if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(publication[blockIdx.x*2+1]));
}

std::vector<Tensor> route_projected(Tensor xy,Tensor radii,Tensor order,Tensor assignment,int tw,int th,int ts,
    Tensor desc,Tensor bitmap,Tensor publication){
  c10::cuda::CUDAGuard guard(xy.device());int N=order.size(1),blocks=(N+255)/256;
  auto raw=torch::empty({N,8},radii.options());
  if(N)fused_route_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>((float2*)xy.data_ptr<float>(),
    (int2*)radii.data_ptr<int>(),order.data_ptr<int64_t>(),assignment.data_ptr<int64_t>(),raw.data_ptr<int>(),
    desc.data_ptr<int>(),bitmap.data_ptr<int>(),(unsigned long long*)publication.data_ptr<int64_t>(),
    N,desc.size(1),bitmap.size(1),(tw+7)/8,(th+3)/4,xy.size(1),xy.size(0),tw,th,ts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {desc.narrow(1,0,N),bitmap.narrow(1,0,blocks*8),publication.narrow(0,0,blocks),raw};
}


__global__ void wait_dies_kernel(const unsigned long long* publication,const int4* directory,
    const unsigned long long* delay,int tasks){
  int die=blockIdx.x,lane=threadIdx.x;unsigned long long anchor=0;
  for(int i=lane;i<tasks;i+=32)if(directory[i].x==die)anchor=max(anchor,publication[i*2+1]);
  for(int shift=16;shift;shift/=2)anchor=max(anchor,__shfl_down_sync(0xffffffff,anchor,shift));
  if(!lane){unsigned long long now,due=anchor+delay[die];
    do{asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));if(now<due)__nanosleep(64);}while(now<due);}
}
void wait_dies(Tensor publication,Tensor directory,Tensor delay){
  c10::cuda::CUDAGuard guard(publication.device());
  wait_dies_kernel<<<32,32,0,at::cuda::getCurrentCUDAStream()>>>((unsigned long long*)publication.data_ptr<int64_t>(),
    (int4*)directory.data_ptr<int>(),(unsigned long long*)delay.data_ptr<int64_t>(),directory.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void jobs_kernel(int64_t* jobs,int tiles){int i=blockIdx.x*256+threadIdx.x;if(i<tiles)jobs[i]=i;}
__global__ void stamp_kernel(unsigned long long* time){asm volatile("mov.u64 %0, %%globaltimer;":"=l"(time[0]));}
std::tuple<Tensor,Tensor,std::vector<int64_t>,Tensor> ranges(Tensor counts){
  c10::cuda::CUDAGuard guard(counts.device());int tiles=counts.numel()-1;
  auto offsets=torch::empty_like(counts),jobs=torch::empty({tiles},counts.options().dtype(torch::kInt64));
  auto reads=torch::empty({1},counts.options().dtype(torch::kInt64));
  size_t bytes=0;auto stream=at::cuda::getCurrentCUDAStream();
  stamp_kernel<<<1,1,0,stream>>>((unsigned long long*)reads.data_ptr<int64_t>());
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,bytes,counts.data_ptr<int>(),offsets.data_ptr<int>(),tiles+1,stream));
  auto tmp=torch::empty({int64_t(bytes)},counts.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(tmp.data_ptr(),bytes,counts.data_ptr<int>(),offsets.data_ptr<int>(),tiles+1,stream));
  jobs_kernel<<<(tiles+255)/256,256,0,stream>>>(jobs.data_ptr<int64_t>(),tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {jobs,offsets,{tiles},reads};
}
__global__ void scatter_tile_ids(const int* input,const int2* source,const int* offsets,
    int capacity,int* output,int slots,unsigned long long* reads){
  int tile=blockIdx.x;int2 location=source[tile];int first=offsets[tile],count=offsets[tile+1]-first;
  if(!threadIdx.x)asm volatile("mov.u64 %0, %%globaltimer;":"=l"(reads[tile]));
  for(int j=threadIdx.x;j<count;j+=blockDim.x){
    int i=first+j,block=i/256,owner=(block%32)*12+(block/32)%12,slot=(block/384)*256+i%256;
    wt(output+owner*slots+slot,input[location.x*capacity+location.y+j]);
  }
}
std::vector<Tensor> scatter_ids(Tensor input,Tensor locations,Tensor offsets,int n){
  c10::cuda::CUDAGuard guard(input.device());int tiles=locations.size(0),slots=((n+98303)/98304)*256;
  auto output=torch::empty({384,slots},input.options());
  auto reads=torch::empty({tiles},input.options().dtype(torch::kInt64));
  if(n)scatter_tile_ids<<<tiles,128,0,at::cuda::getCurrentCUDAStream()>>>(input.data_ptr<int>(),(int2*)locations.data_ptr<int>(),
    offsets.data_ptr<int>(),input.size(1),output.data_ptr<int>(),slots,(unsigned long long*)reads.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,reads};
}

__global__ void spatial_counts(const int* metadata,const unsigned* bitmap,int* counts,int N,int words,int meta_stride){
  int i=blockIdx.x*256+threadIdx.x,die=blockIdx.y;if(i>=N)return;
  bool live=(bitmap[die*words+i/32]>>(i%32))&1u;
  const int* r=metadata+die*meta_stride+i*8;
  counts[die*N+i]=live?(r[2]-r[1])*(r[4]-r[3]):0;
}
__global__ void expand_spatial(const int* metadata,const unsigned* bitmap,const int* offsets,
    int2* output,int N,int words,int capacity,int tw,int meta_stride){
  int i=blockIdx.x*256+threadIdx.x,die=blockIdx.y;if(i>=N)return;
  if(!((bitmap[die*words+i/32]>>(i%32))&1u))return;
  const int* r=metadata+die*meta_stride+i*8;int first=offsets[die*N+i]-offsets[die*N];
  for(int y=r[3];y<r[4];++y)for(int x=r[1];x<r[2];++x){
    int key=y*tw+x,g=r[0];
    asm volatile("st.global.wt.v2.u32 [%0], {%1,%2};"::"l"(output+die*capacity+first),"r"(key),"r"(g):"memory");++first;
  }
}
Tensor expand_routes(Tensor metadata,Tensor bitmap,Tensor output,int tw){
  c10::cuda::CUDAGuard guard(metadata.device());int N=metadata.size(1);
  auto counts=torch::empty({32*N},metadata.options()),offsets=torch::empty_like(counts);
  auto stream=at::cuda::getCurrentCUDAStream();
  spatial_counts<<<dim3((N+255)/256,32),256,0,stream>>>(metadata.data_ptr<int>(),(unsigned*)bitmap.data_ptr<int>(),counts.data_ptr<int>(),N,bitmap.stride(0),metadata.stride(0));
  size_t bytes=0;C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr,bytes,counts.data_ptr<int>(),offsets.data_ptr<int>(),32*N,stream));
  auto temporary=torch::empty({int64_t(bytes)},metadata.options().dtype(torch::kUInt8));
  C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temporary.data_ptr(),bytes,counts.data_ptr<int>(),offsets.data_ptr<int>(),32*N,stream));
  expand_spatial<<<dim3((N+255)/256,32),256,0,stream>>>(metadata.data_ptr<int>(),(unsigned*)bitmap.data_ptr<int>(),offsets.data_ptr<int>(),
    (int2*)output.data_ptr<int>(),N,bitmap.stride(0),output.size(1),tw,metadata.stride(0));
  auto time=torch::empty({1},metadata.options().dtype(torch::kInt64));
  // A single completion event covers the actual GPU output writes.
  stamp_kernel<<<1,1,0,stream>>>((unsigned long long*)time.data_ptr<int64_t>());C10_CUDA_KERNEL_LAUNCH_CHECK();return time;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("expand_routes",&expand_routes);m.def("ranges",&ranges);m.def("scatter_ids",&scatter_ids);m.def("wait_dies",&wait_dies);m.def("route_projected",&route_projected);m.def("route",&route);m.def("route_into",&route_into);}
