// Bank-local column operations added to the LPDDR5 device specification.
// External RD/WR timing remains unchanged. Local columns use bank timing and
// leave the external data bus free; a configurable die fabric limits gathers.
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <array>
#include <chrono>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "ramulator/base/config.h"
#include "ramulator/dram/device.h"

using namespace Ramulator;
namespace {
struct Input {
  uint64_t address, ready;
  uint32_t kind, depends, cost, tag;
};
struct Stats {
  uint64_t cycles, commands, refreshes, pauses, checks, local_reads, local_writes, gpu_reads, gpu_writes;
  double wall_ms;
};
thread_local std::string error;
thread_local uint64_t sync_issued=0, sync_reused=0;
constexpr uint32_t None = 0xffffffffu;
Clk_t earliest(const DRAMNode* node, int command, const AddrVec_t& address) {
  Clk_t result = std::max<Clk_t>(0, node->m_cmd_ready_clk[command]);
  if (node->m_child_nodes.empty()) return result;
  int id = address[node->m_level + 1];
  if (id < 0) {
    for (const auto& child : node->m_child_nodes)
      result = std::max(result, earliest(child.get(), command, address));
  } else result = std::max(result, earliest(node->m_child_nodes[id].get(), command, address));
  return result;
}

void add_local_columns(DRAMSpec& spec) {
  const int old_count = spec.command_count, bank = spec.get_level_id("Bank");
  const int rd = spec.get_command_id("RD"), wr = spec.get_command_id("WR");
  spec.command_count += 2;
  spec.commands["PIM_RD"] = old_count; spec.commands["PIM_WR"] = old_count + 1;
  spec.command_names.push_back("PIM_RD"); spec.command_names.push_back("PIM_WR");
  spec.funcs.resize(spec.command_count);
  for (int original : {rd, wr}) {
    const int added = old_count + (original == wr);
    spec.command_cycles.push_back(1);
    spec.command_meta.push_back(spec.command_meta[original]);
    spec.bank_targets.push_back(BankTarget::Single);
    spec.pair_rules.push_back(spec.pair_rules[original]);
    spec.funcs.preqs[added] = spec.funcs.preqs[original];
    spec.funcs.actions[added] = spec.funcs.actions[original];
    spec.funcs.rowhits[added] = spec.funcs.rowhits[original];
    spec.funcs.rowopens[added] = spec.funcs.rowopens[original];
  }
  for (int level = 0; level < spec.level_count; ++level) {
    const auto old = spec.timing_cons[level];
    spec.timing_cons[level].resize(spec.command_count);
    spec.has_sibling_cons[level].resize(spec.command_count, false);
    for (int cmd = 0; cmd < old_count; ++cmd) {
      for (auto constraint : old[cmd]) {
        // A local access obeys row activation/precharge/refresh timing and
        // same-bank array timing. External bus and rank-switch rules do not
        // constrain bank-internal transfers.
        bool local_source = cmd == rd || cmd == wr;
        bool local_target = constraint.cmd == rd || constraint.cmd == wr;
        auto local = [&](int c) { return old_count + (c == wr); };
        if (local_target && (level == bank || (!spec.command_meta[cmd].is_accessing && level > 0))) {
          auto copy = constraint; copy.cmd = local(constraint.cmd);
          spec.timing_cons[level][cmd].push_back(copy);
        }
        if (local_source && level == bank) {
          spec.timing_cons[level][local(cmd)].push_back(constraint);
          if (local_target) {
            constraint.cmd = local(constraint.cmd);
            spec.timing_cons[level][local(cmd)].push_back(constraint);
          }
        }
        if (local_source && level == spec.get_level_id("Rank") && !local_target &&
            (spec.command_names[constraint.cmd] == "PREab" || spec.command_names[constraint.cmd] == "REFab"))
          spec.timing_cons[level][local(cmd)].push_back(constraint);
      }
    }
  }
  // The array cannot perform two column transfers in the same cycle even
  // when DQ is bypassed. Use same-bank tCCD and direction turnarounds.
  int lr = old_count, lw = old_count + 1;
  auto add = [&](int a, int b, int value) { spec.timing_cons[bank][a].emplace_back(b, value); };
  const int ccd = spec.get_timing_value("nCCDL");
  const int rtow = spec.get_timing_value("nCL") + spec.get_timing_value("nBL_max") + 2 - spec.get_timing_value("nCWL");
  const int wtor = spec.get_timing_value("nCWL") + spec.get_timing_value("nBL_max") + spec.get_timing_value("nWTRL");
  for (int a : {rd, lr}) for (int b : {rd, lr}) add(a,b,ccd);
  for (int a : {wr, lw}) for (int b : {wr, lw}) add(a,b,ccd);
  for (int a : {rd, lr}) for (int b : {wr, lw}) add(a,b,rtow);
  for (int a : {wr, lw}) for (int b : {rd, lr}) add(a,b,wtor);
}

struct Job : Input {
  uint64_t index;
  AddrVec_t address;
  bool sync = false;
  int flat_bank=0;
  int priority_offset=0;
};
struct FenceRanges {
  static constexpr uint64_t block_size=128;
  static constexpr uint64_t unavailable=std::numeric_limits<uint64_t>::max();
  uint64_t count, leaves=1;
  const std::vector<uint8_t>& issued;
  const uint64_t* completion;
  std::vector<uint16_t> remaining;
  std::vector<uint64_t> maxima, tree;
  FenceRanges(uint64_t n,const std::vector<uint8_t>& i,const uint64_t* c)
      :count(n),issued(i),completion(c),remaining((n+block_size-1)/block_size),maxima(remaining.size(),0) {
    while(leaves<remaining.size())leaves*=2;
    tree.assign(leaves*2,unavailable);
    for(uint64_t b=0;b<remaining.size();++b)remaining[b]=std::min(block_size,n-b*block_size);
  }
  void issue(uint64_t index) {
    auto block=index/block_size;
    maxima[block]=std::max(maxima[block],completion[index]);
    if(--remaining[block])return;
    auto node=leaves+block;tree[node]=maxima[block];
    while(node>1){node/=2;tree[node]=std::max(tree[node*2],tree[node*2+1]);}
  }
  uint64_t query(uint64_t first,uint64_t last) const {
    uint64_t result=0;
    while(first<last&&first%block_size) {
      if(!issued[first])return unavailable;
      result=std::max(result,completion[first++]);
    }
    while(first<last&&last%block_size) {
      if(!issued[--last])return unavailable;
      result=std::max(result,completion[last]);
    }
    auto a=leaves+first/block_size,b=leaves+last/block_size;
    while(a<b) {
      if(a&1)result=std::max(result,tree[a++]);
      if(b&1)result=std::max(result,tree[--b]);
      if(result==unavailable)return result;
      a/=2;b/=2;
    }
    return result;
  }
};
struct Execution {
  struct promise_type {
    int reason=0;
    std::exception_ptr failure;
    Execution get_return_object() {return Execution{std::coroutine_handle<promise_type>::from_promise(*this)};}
    std::suspend_always initial_suspend() noexcept {return {};}
    std::suspend_always final_suspend() noexcept {return {};}
    std::suspend_always yield_value(int value) noexcept {reason=value;return {};}
    void return_void() noexcept {reason=0;}
    void unhandled_exception() noexcept {failure=std::current_exception();}
  };
  std::coroutine_handle<promise_type> handle;
  explicit Execution(std::coroutine_handle<promise_type> value):handle(value) {}
  Execution(const Execution&)=delete;
  Execution(Execution&& other) noexcept:handle(std::exchange(other.handle,{})) {}
  ~Execution() {if(handle)handle.destroy();}
  int advance() {
    if(!handle.done())handle.resume();
    if(handle.promise().failure)std::rethrow_exception(handle.promise().failure);
    return handle.done()?0:handle.promise().reason;
  }
};
struct PendingSlots {
  std::vector<int> next, previous;
  int first=-1, last=-1;
  size_t count=0;
  void push(int slot) {
    if(size_t(slot)>=next.size()){next.resize(slot+1);previous.resize(slot+1);}
    previous[slot]=last;next[slot]=-1;
    if(last<0)first=slot;else next[last]=slot;
    last=slot;++count;
  }
  void erase(int slot) {
    if(previous[slot]<0)first=next[slot];else next[previous[slot]]=next[slot];
    if(next[slot]<0)last=previous[slot];else previous[next[slot]]=previous[slot];
    --count;
  }
};
struct Replay {
  DRAMDevice device;
  Stats stats{};
  Clk_t clk=0, ca=0, next_ref=0, drain=0;
  std::vector<Clk_t> compute, fabric, hcf_alu;
  std::vector<Clk_t> wck_expiry;
  std::vector<std::array<DRAMNode*,4>> bank_timing_paths;
  uint64_t cas_owner=std::numeric_limits<uint64_t>::max();
  int rd,wr,lr,lw,casr,casw,ref;
  Clk_t refi,rfc,latency,write_latency;
  Clk_t read_command_latency,write_command_latency,burst_cycles,wck_postamble;
  bool refresh, jump;
  std::vector<Clk_t> fragment_ready;
  std::vector<uint64_t> fragment_owner;
  uint32_t fragment_capacity;
  uint32_t staging_bytes=0;
  std::vector<uint32_t> fragment_bytes;
  int ranks, fragment_slots;
  int request_capacity=32;
  bool parallel_alu=false, row_hit_first=false;
  bool actual_submissions=false, bank_submissions=false;
  std::vector<uint64_t> published_inputs;
  Clk_t horizon=std::numeric_limits<Clk_t>::max();
  std::vector<uint64_t> publications;
  uint64_t unpublished=0;
  std::vector<uint32_t> watch_head,watch_next,watch_group,watch_remaining;
  std::vector<uint64_t> watch_maximum;
  std::vector<std::array<uint64_t,2>> notifications;
  void observe(uint64_t index,uint64_t completion) {
    if(watch_head.empty())return;
    for(auto edge=watch_head[index];edge!=None;edge=watch_next[edge]) {
      auto group=watch_group[edge];
      watch_maximum[group]=std::max(watch_maximum[group],completion);
      if(!--watch_remaining[group])notifications.push_back({group,watch_maximum[group]});
    }
  }
  bool refresh_active=false;
  int refresh_rank=0;
  Clk_t refresh_started=0,refresh_until=0;
  explicit Replay(const char* config, bool refresh_, bool jump_, int slots=2, int capacity=8192, int ranks_=1)
      : refresh(refresh_),jump(jump_),fragment_ready(slots*ranks_,0),
        fragment_owner(slots*ranks_,std::numeric_limits<uint64_t>::max()),fragment_capacity(capacity),
        ranks(ranks_),fragment_slots(slots) {
    if(slots<=0||capacity<288||ranks<1||ranks>2)throw std::runtime_error("invalid HCF fragment pool or channel ranks");
    if(const char* value=std::getenv("PIM_ROW_HIT_FIRST"))row_hit_first=std::stoi(value)!=0;
    if(const char* value=std::getenv("PIM_ACTUAL_GPU_SUBMISSIONS"))actual_submissions=std::stoi(value)!=0;
    if(const char* value=std::getenv("PIM_BANK_SUBMISSIONS"))bank_submissions=std::stoi(value)!=0;
    if(const char* value=std::getenv("PIM_PARALLEL_ALU"))parallel_alu=std::stoi(value)!=0;
    if(const char* value=std::getenv("PIM_STAGING_BYTES")) {
      staging_bytes=std::stoi(value);
      if(staging_bytes!=16384)throw std::runtime_error("suballocated staging must retain 16 KiB per die");
      fragment_slots=64;fragment_capacity=staging_bytes;
      fragment_ready.assign(fragment_slots*ranks,0);
      fragment_owner.assign(fragment_slots*ranks,std::numeric_limits<uint64_t>::max());
      fragment_bytes.assign(fragment_slots*ranks,0);
    }
    auto cfg=Config::parse_config_file(config);
    if(ranks!=1) {
      auto dram=cfg["dram"],org=dram["org"];
      auto counts=org["count"].seq();counts[1]=ConfigNode(ranks);
      org.set("count",ConfigNode(counts));dram.set("org",org);cfg.set("dram",dram);
    }
    auto spec=DRAMSpec::create("LPDDR5",cfg);
    add_local_columns(*spec);
    device.init(std::move(spec));
    auto& s=*device.m_spec;
    if(s.get_level_id("Bank")!=3)throw std::runtime_error("flat timing path requires four LPDDR5 hierarchy levels");
    for(auto* bank:device.m_bank_nodes) {
      std::array<DRAMNode*,4> path{};
      for(auto* node=bank;node;node=node->m_parent_node)path[node->m_level]=node;
      bank_timing_paths.push_back(path);
    }
    rd=s.get_command_id("RD");wr=s.get_command_id("WR");
    lr=s.get_command_id("PIM_RD");lw=s.get_command_id("PIM_WR");
    casr=s.get_command_id("CAS_RD");casw=s.get_command_id("CAS_WR");ref=s.get_command_id("REFab");
    refi=s.get_timing_value("nREFI");rfc=s.get_timing_value("nRFC");
    latency=s.read_latency;
    write_latency=s.get_timing_value("nCWL")+s.get_timing_value("nBL_min")+s.get_timing_value("nWR")+1;
    read_command_latency=s.get_timing_value("nCL");
    write_command_latency=s.get_timing_value("nCWL");
    burst_cycles=s.get_timing_value("nBL_max");
    wck_postamble=s.get_timing_value("nWCKPST");
    wck_expiry.resize(ranks,0);
    sync_issued=sync_reused=0;
    if(const char* value=std::getenv("PIM_MEMORY_WINDOW")) {
      request_capacity=std::stoi(value);
      if(request_capacity<32||request_capacity>256)throw std::runtime_error("invalid request capacity");
    }
    compute.resize(s.get_level_size("Rank")*16,0);
    fabric.resize(ranks,0);hcf_alu.resize(ranks,0);
    next_ref=refi;
  }
  int rank_bank(const AddrVec_t& a) const {return (a[1]*4+a[2])*4+a[3];}
  Clk_t job_earliest(int command,int bank) const {
    const auto& p=bank_timing_paths[bank];
    return std::max<Clk_t>({0,p[0]->m_cmd_ready_clk[command],p[1]->m_cmd_ready_clk[command],
                           p[2]->m_cmd_ready_clk[command],p[3]->m_cmd_ready_clk[command]});
  }
  void issue(int command,const AddrVec_t& a) {
    if(device.get_preq_command(command,a,clk)!=command || !device.check_timing(command,a,clk))
      throw std::runtime_error("illegal joint command");
    device.issue_command(command,a,clk);++stats.commands;ca=clk+1;
  }
  bool service_refresh() {
    if(!refresh||(!refresh_active&&clk<next_ref))return true;
    if(!refresh_active) {
      refresh_active=true;refresh_rank=0;refresh_started=clk;
    }
    Clk_t target_clk;
    if(refresh_rank<ranks) {
      target_clk=std::max({clk,ca,drain,*std::max_element(compute.begin(),compute.end()),*std::max_element(hcf_alu.begin(),hcf_alu.end())});
      if(target_clk>=horizon){clk=horizon;return false;}
      clk=target_clk;
    }
    while(refresh_rank<ranks) {
      AddrVec_t a={0,refresh_rank,-1,-1,-1,-1};
      int c=device.get_preq_command(ref,a,clk);
      auto target=device.preq_addr_vec(ref,c,a,clk);
      target_clk=std::max({clk,ca,earliest(device.m_root.get(),c,target)});
      if(target_clk>=horizon){clk=horizon;return false;}
      clk=target_clk;
      issue(c,target);
      if(c==ref) {
        ++refresh_rank;
        if(refresh_rank<ranks)clk=ca;
        else {refresh_until=clk+rfc;break;}
      } else clk=ca;
      }
    // Once REFab has issued, no ordinary command can run during its busy
    // interval. Keep that interval across externally imposed horizons.
    if(refresh_until>=horizon){clk=horizon;return false;}
    clk=refresh_until;
    next_ref+=refi;stats.refreshes+=ranks;stats.pauses+=clk-refresh_started;
    refresh_active=false;
    return true;
  }
  Execution execute(const Input* in,uint64_t count,uint64_t* completion,bool streams=false) {
    int priority_spans=0;
    if(const char* value=std::getenv("PIM_WINDOW_PRIORITY")) {
      priority_spans=std::stoi(value);
      if(priority_spans<1||10*priority_spans>256)throw std::runtime_error("invalid direct-pack window namespace");
    }
    std::vector<unsigned> span_windows;
    if(const char* value=std::getenv("PIM_SPAN_WINDOWS")) {
      std::string windows(value);size_t first=0;
      while(first<windows.size()) {
        size_t last=windows.find(',',first);
        int window=std::stoi(windows.substr(first,last-first));
        if(window<0||window>=128)throw std::runtime_error("invalid span window");
        span_windows.push_back(unsigned(window));
        if(last==std::string::npos)break;
        first=last+1;
      }
    }
    if(priority_spans&&span_windows.size()!=size_t(priority_spans))
      throw std::runtime_error("each direct-pack span requires a window mapping");
    unsigned priority_windows=span_windows.empty()?0:
        *std::max_element(span_windows.begin(),span_windows.end())+1;
    int auxiliary_lane=256;
    if(const char* value=std::getenv("PIM_AUXILIARY_LANE_START")) {
      auxiliary_lane=std::stoi(value);
      if(!priority_spans||auxiliary_lane<10*priority_spans||auxiliary_lane>=256)
        throw std::runtime_error("invalid auxiliary stream namespace");
    }
    const char* ready_value=std::getenv("PIM_READY_DEPENDENCIES");
    bool ready_dependencies=ready_value&&std::stoi(ready_value)!=0;
    std::array<bool,256> bank_field_lanes{};
    if(const char* value=std::getenv("PIM_FIELD_BANK_LANES")) {
      std::string lanes(value);size_t first=0;
      while(first<lanes.size()) {
        size_t last=lanes.find(',',first);
        int lane=std::stoi(lanes.substr(first,last-first));
        if(lane<0||lane>=256)throw std::runtime_error("invalid field bank lane");
        bank_field_lanes[lane]=true;
        if(last==std::string::npos)break;
        first=last+1;
      }
    }
    for(uint64_t i=0;i<count;++i) {
      if(in[i].depends!=None && in[i].depends>=i)throw std::runtime_error("dependency must point to earlier request");
      if(in[i].kind>10)throw std::runtime_error("unknown request kind");
      if(in[i].kind==7&&in[i].cost>fragment_capacity)throw std::runtime_error("HCF fragment exceeds capacity");
      if(in[i].kind==8&&(in[i].cost>=i||in[in[i].cost].kind!=7))throw std::runtime_error("invalid HCF release");
      int rank=in[i].address>>63;
      if(rank>=ranks)throw std::runtime_error("address rank is not attached to this channel");
    }
    std::vector<uint8_t> issued(count,0), hcf_slots(count,255);
    FenceRanges fences(count,issued,completion);
    std::array<std::vector<uint64_t>,512> lanes;
    if(streams)for(uint64_t i=0;i<count;++i)
      lanes[((in[i].address>>63)<<8)|(in[i].tag>>24)].push_back(i);
    std::vector<uint32_t> submitted_reads;
    std::deque<uint64_t> submitted;
    std::array<std::deque<uint64_t>,32> bank_submitted;
    unsigned submit_turn=0;
    if(actual_submissions)submitted_reads.assign(count,None);
    if(const char* order=std::getenv("PIM_GPU_SOURCE_ORDER");order&&std::stoi(order)) {
      // Undispatched GPU CTAs have not submitted requests. Keep the captured
      // per-bank dispatch order while retaining the finite channel window.
      for(auto& indices:lanes) {
        if(actual_submissions) {
          bool source=false,only_source=true;
          for(size_t at=0;at<indices.size();++at) {
            auto index=indices[at];const auto& input=in[index];
            if(input.kind==10&&input.ready>0&&at+1<indices.size()&&
               in[indices[at+1]].kind==0&&in[indices[at+1]].depends==index) {
              source=true;++at;
            } else if(input.kind==0&&input.ready>0)source=true;
            else if(!(input.kind==6&&input.ready==0)){only_source=false;break;}
          }
          if(!source||!only_source)continue;
          size_t retained=0;
          for(size_t at=0;at<indices.size();++at) {
            auto index=indices[at];
            if(in[index].kind==10)submitted_reads[index]=indices[++at];
            else if(in[index].kind==6)indices[retained++]=index;
            else throw std::runtime_error("actual GPU submission needs external event/read pairs");
          }
          indices.resize(retained);continue;
        }
        struct SourceGroup {unsigned window,tie;uint64_t ready;std::vector<uint64_t> indices;};
        std::vector<SourceGroup> groups;
        bool source=false,only_source=true;
        for(size_t at=0;at<indices.size();++at) {
          auto index=indices[at];auto& input=in[index];
          SourceGroup group{(input.tag>>16)&255,input.tag&65535,input.ready,{index}};
          if(input.kind==10&&input.ready>0&&at+1<indices.size()&&
             in[indices[at+1]].kind==0&&in[indices[at+1]].depends==index) {
            source=true;group.indices.push_back(indices[++at]);
          } else if(input.kind==0&&input.ready>0)source=true;
          else if(!(input.kind==6&&input.ready==0)) {only_source=false;break;}
          groups.push_back(std::move(group));
        }
        if(!source||!only_source)continue;
        std::stable_sort(groups.begin(),groups.end(),[&](auto& a,auto& b) {
          return a.window!=b.window?a.window<b.window:
                 a.ready!=b.ready?a.ready<b.ready:a.tie<b.tie;
        });
        indices.clear();
        for(auto& group:groups)indices.insert(indices.end(),group.indices.begin(),group.indices.end());
      }
    }
    struct Lane {std::vector<uint64_t> indices;size_t head=0;};
    std::vector<Lane> ordered_lanes;ordered_lanes.reserve(lanes.size());
    for(auto& indices:lanes)if(!indices.empty())ordered_lanes.push_back({std::move(indices),0});
    std::vector<size_t> active_lanes;
    for(size_t i=0;i<ordered_lanes.size();++i)
      if(!ordered_lanes[i].indices.empty())active_lanes.push_back(i);
    size_t lane_turn=0;
    uint64_t retired=0,head=0;
    size_t memory_pending=0;
    std::vector<Job> jobs;jobs.reserve(request_capacity);
    PendingSlots pending, logical;
    std::vector<size_t> vacant;
    std::vector<uint64_t> admission_order;
    uint64_t admitted=0;
    std::set<std::pair<uint64_t,int>> ready_publications;
    std::unordered_map<uint64_t,int> pending_publications;
    std::unordered_multimap<uint64_t,int> publication_dependencies;
    size_t processed_publications=0;
    auto activate_publication=[&](int slot) {
      const auto& job=jobs[slot];
      if(!publications.empty()&&publications[job.index]!=FenceRanges::unavailable&&
         (job.depends==None||issued[job.depends]))
        ready_publications.emplace(admission_order[slot],slot);
    };
    std::vector<AddrVec_t> address_pool(request_capacity,AddrVec_t(6));
    auto request_ready=[&](const Input& input) {return ready_dependencies&&input.kind==10?uint64_t(0):input.ready;};
    auto admit=[&](uint64_t index) {
      if(address_pool.empty())address_pool.emplace_back(6);
      Job job{in[index],index,std::move(address_pool.back())};address_pool.pop_back();
      job.ready=request_ready(in[index]);
      if(priority_spans) {
        unsigned lane=(job.tag>>24)&255;
        bool bank_field=bank_field_lanes[lane]&&(job.tag&(1u<<23));
        if(bank_field||lane>=unsigned(auxiliary_lane)) {
          unsigned window=(job.tag>>16)&(bank_field?127:255);
          if(window>=priority_windows)throw std::runtime_error("auxiliary stream window outside capture");
          job.priority_offset=window*8;
        } else {
          unsigned span=lane<unsigned(2*priority_spans)?lane/2:
              lane<unsigned(6*priority_spans)?(lane-2*priority_spans)/4:(lane-6*priority_spans)/4;
          if(span>=unsigned(priority_spans))throw std::runtime_error("direct-pack stream outside window namespace");
          job.priority_offset=span_windows[span]*8;
        }
      }
      auto& vector=job.address;uint64_t a=(in[index].address&0x7fffffffffffffffULL)/32;
      vector[0]=0;vector[1]=in[index].address>>63;vector[5]=(a%64)*16;a/=64;
      int bank=a%16;a/=16;vector[2]=bank/4;vector[3]=bank%4;vector[4]=int(a);
      job.flat_bank=rank_bank(vector);
      if(job.kind<4)++memory_pending;
      size_t slot;
      if(vacant.empty()){slot=jobs.size();jobs.push_back(std::move(job));}
      else {slot=vacant.back();vacant.pop_back();jobs[slot]=std::move(job);}
      pending.push(slot);
      if(slot>=admission_order.size())admission_order.resize(slot+1);
      admission_order[slot]=admitted++;
      if(in[index].kind>=5&&in[index].kind!=9)logical.push(slot);
      if(in[index].kind==10) {
        pending_publications.emplace(index,slot);
        if(in[index].depends!=None&&!issued[in[index].depends])
          publication_dependencies.emplace(in[index].depends,slot);
        activate_publication(slot);
      }
    };
    auto retire=[&](int slot) {
      auto& job=jobs[slot];auto index=job.index;
      if(job.kind<4)--memory_pending;
      fences.issue(index);
      stats.cycles=std::max(stats.cycles,completion[index]);
      observe(index,completion[index]);
      address_pool.push_back(std::move(job.address));
      if(job.kind==10) {
        pending_publications.erase(index);
        ready_publications.erase({admission_order[slot],slot});
      }
      auto [first,last]=publication_dependencies.equal_range(index);
      for(auto it=first;it!=last;++it)activate_publication(it->second);
      publication_dependencies.erase(first,last);
      if(job.kind>=5&&job.kind!=9)logical.erase(slot);
      vacant.push_back(slot);pending.erase(slot);++retired;
    };
    while(retired<count) {
      if(clk>=horizon){co_yield 1;continue;}
      while(processed_publications<published_inputs.size()) {
        auto index=published_inputs[processed_publications++];
        auto found=pending_publications.find(index);
        if(found!=pending_publications.end())activate_publication(found->second);
      }
      if(!ready_publications.empty()) {
        auto slot=ready_publications.begin()->second;
        auto& j=jobs[slot];auto index=j.index;
        completion[index]=std::max({j.ready,publications[index],j.depends==None?0:completion[j.depends]});
        issued[index]=true;retire(slot);continue;
      }
      // A separate CAS reserves the following command slot for its RD/WR.
      if(cas_owner==std::numeric_limits<uint64_t>::max()&&!service_refresh()) {
        co_yield 1;continue;
      }
      Clk_t next=std::numeric_limits<Clk_t>::max();
      if(!streams)while(head<count&&pending.count<size_t(request_capacity))admit(head++);
      else {
        // Memory requests share a bounded window. Dependency bookkeeping
        // must remain serviceable so staging releases can drain that window.
        if(actual_submissions) {
          for(auto index:published_inputs)if(submitted_reads[index]!=None) {
            if(bank_submissions) {
              uint64_t encoded=in[submitted_reads[index]].address;
              unsigned bank=((encoded&0x7fffffffffffffffULL)/2048)%16+16*(encoded>>63);
              bank_submitted[bank].push_back(index);
              bank_submitted[bank].push_back(submitted_reads[index]);
            } else {submitted.push_back(index);submitted.push_back(submitted_reads[index]);}
          }
          published_inputs.clear();processed_publications=0;
        }
        bool added=true;
        while(added) {
          added=false;
          if(bank_submissions) {
            for(unsigned step=0;step<bank_submitted.size();++step) {
              unsigned bank=(submit_turn+step)%bank_submitted.size();
              auto& queue=bank_submitted[bank];
              if(queue.empty())continue;
              auto index=queue.front();const auto& j=in[index];
              if((j.kind>=4||memory_pending<size_t(request_capacity))&&
                 (j.depends==None||issued[j.depends])) {
                admit(index);queue.pop_front();added=true;
                submit_turn=(bank+1)%bank_submitted.size();break;
              }
            }
          } else if(!submitted.empty()) {
            auto index=submitted.front();const auto& j=in[index];
            if((j.kind>=4||memory_pending<size_t(request_capacity))&&
               (j.depends==None||issued[j.depends])) {
              admit(index);submitted.pop_front();added=true;
            }
          }
          for(size_t step=0;step<active_lanes.size();++step) {
            size_t position=(lane_turn+step)%active_lanes.size();
            auto& lane=ordered_lanes[active_lanes[position]];auto& indices=lane.indices;
            auto& at=lane.head;
            if(at==indices.size())continue;
            uint64_t index=indices[at];const auto& j=in[index];
            if(j.kind<4&&memory_pending>=size_t(request_capacity))continue;
            if(j.depends!=None&&!issued[j.depends])continue;
            Clk_t release=std::max<Clk_t>(request_ready(j),j.depends==None?0:completion[j.depends]);
            if(j.kind==5) {
              if(fences.query(j.cost,index)==FenceRanges::unavailable)continue;
            }
            if((j.kind<5||j.kind==9)&&release>clk){next=std::min(next,release);continue;}
            admit(index);++at;added=true;
            if(at==indices.size()) {
              active_lanes.erase(active_lanes.begin()+position);
              lane_turn=active_lanes.empty()?0:position%active_lanes.size();
            } else lane_turn=(position+1)%active_lanes.size();
            break;
          }
        }
      }
      int selected=-1,command=-1,priority=99;
      for(int slot=logical.first;slot!=-1;slot=logical.next[slot]) {
        auto& j=jobs[slot];
        if(j.depends!=None && !issued[j.depends])continue;
        Clk_t ready=std::max<Clk_t>(j.ready,j.depends==None?0:completion[j.depends]);
        if(j.kind==10) {
          auto index=jobs[slot].index;
          if(publications.empty()||publications[index]==FenceRanges::unavailable)continue;
          ready=std::max<Clk_t>(ready,publications[index]);
          issued[j.index]=true;completion[index]=ready;
          retire(slot);selected=-2;break;
        }
        if(j.kind==7) {
          auto begin=fragment_ready.begin()+j.address[1]*fragment_slots;
          auto available=std::min_element(begin,begin+fragment_slots);
          if(staging_bytes) {
            uint32_t live=0;
            for(int i=j.address[1]*fragment_slots;i<(j.address[1]+1)*fragment_slots;++i) {
              if(fragment_owner[i]!=std::numeric_limits<uint64_t>::max()||fragment_ready[i]>clk)live+=fragment_bytes[i];
            }
            uint32_t bytes=((j.cost+255)/256)*256;
            if(live+bytes>staging_bytes) {
              for(auto it=begin;it!=begin+fragment_slots;++it)
                if(*it>clk&&*it!=std::numeric_limits<Clk_t>::max())next=std::min(next,*it);
              continue;
            }
          }
          if(*available==std::numeric_limits<Clk_t>::max())continue;
          Clk_t acquire_at=std::max(ready,*available);
          if(acquire_at>clk){next=std::min(next,acquire_at);continue;}
          Clk_t prior_drain=*available;
          int id=available-fragment_ready.begin();
          hcf_slots[j.index]=id;*available=std::numeric_limits<Clk_t>::max();
          fragment_owner[id]=j.index;
          if(staging_bytes)fragment_bytes[id]=((j.cost+255)/256)*256;
          issued[j.index]=true;completion[j.index]=std::max({ready,prior_drain,clk});
          retire(slot);selected=-2;break;
        }
        if(j.kind==8) {
          if(!issued[j.cost])continue;
          int id=hcf_slots[j.cost];
          if(id==255||fragment_owner[id]!=j.cost)throw std::runtime_error("HCF slot ownership mismatch");
          fragment_ready[id]=ready;fragment_owner[id]=std::numeric_limits<uint64_t>::max();
          issued[j.index]=true;completion[j.index]=ready;
          retire(slot);selected=-2;break;
        }
        if(j.kind==6) {
          // A measured GPU consumer releases its ping-pong slot after cost
          // CK. Its already-measured memory service is not charged again.
          // Logical GPU events run independently of the memory controller;
          // controller refresh or polling must not move their timestamps.
          issued[j.index]=true;completion[j.index]=ready+j.cost;
          retire(slot);selected=-2;break;
        }
        if(j.kind==5) {
          auto range=fences.query(j.cost,j.index);
          if(range==FenceRanges::unavailable)continue;
          ready=std::max<Clk_t>(ready,range);
          issued[j.index]=true;completion[j.index]=ready;
          retire(slot);selected=-2;break;
        }
      }
      if(selected==-2)continue;
      for(int slot=pending.first;slot!=-1;slot=pending.next[slot]) {
        auto& j=jobs[slot];
        if(j.kind>=5&&j.kind!=9)continue;
        if(cas_owner!=std::numeric_limits<uint64_t>::max() && j.index!=cas_owner)continue;
        if(j.depends!=None && !issued[j.depends])continue;
        if(priority<99) {
          int best=j.kind==4||j.kind==9?(parallel_alu?-1024:4)+j.priority_offset:
              ((row_hit_first?(j.kind<2?0:1):(j.kind<2?0:2))+j.priority_offset);
          if(best>=priority)continue;
        }
        Clk_t ready=std::max<Clk_t>(j.ready,j.depends==None?0:completion[j.depends]);
        int bank=j.flat_bank;
        ready=std::max(ready,j.kind==9?hcf_alu[j.address[1]]:compute[bank]);
        if(j.kind==4||j.kind==9) {
          if(ready>clk){next=std::min(next,ready);continue;}
          Clk_t duration=std::max<uint32_t>(1,j.cost);
          // PPIM retains instruction position and resumes after refresh.
          if(refresh && clk+duration>=next_ref) {
            next=std::min(next,next_ref);continue;
          }
          // These starts reserve an on-die ALU, without issuing a CA command
          // or advancing clk. Launch ready ALUs before filling the independent
          // command slot; bank/shared-ALU occupancy and refresh still gate them.
          int p=(parallel_alu?-1024:4)+j.priority_offset;
          if(priority>p){selected=slot;command=-1;priority=p;}
          continue;
        }
        int target=j.kind==0?rd:j.kind==1?wr:j.kind==2?lr:lw;
        auto* bank_node=device.m_bank_nodes[j.flat_bank];
        auto hit_fn=device.m_spec->funcs.rowhits[target];
        bool hit=hit_fn&&hit_fn(bank_node,target,j.address,clk);
        int p=(row_hit_first ? (hit?0:2)+(j.kind<2?0:1) : (j.kind<2?0:2)+(hit?0:1))+j.priority_offset;
        if(p>=priority)continue;
        auto preq_fn=device.m_spec->funcs.preqs[target];
        int cmd=preq_fn?preq_fn(bank_node,target,j.address,clk):target;
        if(cmd==target&&j.kind<2&&!j.sync&&clk<wck_expiry[j.address[1]])
          next=std::min(next,std::max(ready,wck_expiry[j.address[1]]));
        if(cmd==target&&j.kind<2&&!j.sync&&clk>=wck_expiry[j.address[1]])cmd=j.kind?casw:casr;
        Clk_t candidate=std::max({clk,ready,ca,job_earliest(cmd,j.flat_bank)});
        if(j.kind>=2 && cmd==target)candidate=std::max(candidate,fabric[j.address[1]]);
        next=std::min(next,candidate);++stats.checks;
        if(candidate>clk)continue;
        if(p<priority){selected=slot;command=cmd;priority=p;}
        // Logical events have already retired; no later command can beat p=0.
        if(p==0)break;
      }
      if(selected==-2)continue;
      if(selected>=0) {
        auto& j=jobs[selected];auto index=j.index;
        if(command==-1) {
          auto bank=j.flat_bank;
          if(j.kind==9)hcf_alu[j.address[1]]=completion[index]=clk+std::max<uint32_t>(1,j.cost);
          else compute[bank]=completion[index]=clk+std::max<uint32_t>(1,j.cost);
          issued[j.index]=true;retire(selected);
          continue;
        }
        issue(command,j.address);
        int target=j.kind==0?rd:j.kind==1?wr:j.kind==2?lr:lw;
        if(command==target) {
          completion[index]=clk+((j.kind==1||j.kind==3)?write_latency:latency);
          drain=std::max<Clk_t>(drain,completion[index]);
          if(j.kind>=2)fabric[j.address[1]]=clk+std::max<uint32_t>(1,j.cost);
          if(j.kind<2) {
            auto latency=j.kind?write_command_latency:read_command_latency;
            auto expiry=clk+device.m_spec->command_cycles[command]-1+latency+burst_cycles+wck_postamble;
            wck_expiry[j.address[1]]=std::max(wck_expiry[j.address[1]],expiry);
            if(!j.sync)++sync_reused;
            cas_owner=std::numeric_limits<uint64_t>::max();
          }
          if(j.kind==0)++stats.gpu_reads;
          if(j.kind==1)++stats.gpu_writes;
          if(j.kind==2)++stats.local_reads;
          if(j.kind==3)++stats.local_writes;
          issued[j.index]=true;retire(selected);
        } else if(command==casr||command==casw) {
          j.sync=true;cas_owner=index;++sync_issued;
        }
        ++clk;
      } else {
        if(streams)for(auto id:active_lanes) {
          const auto& lane=ordered_lanes[id];
          const auto& indices=lane.indices;
          auto at=lane.head;
          if(at==indices.size())continue;
          const auto& j=in[indices[at]];
          if(j.kind==10&&(publications.empty()||publications[indices[at]]==FenceRanges::unavailable))continue;
          if(j.depends==None||issued[j.depends]) {
            Clk_t release=std::max<Clk_t>(request_ready(j),j.depends==None?0:completion[j.depends]);
            if(release>clk)next=std::min(next,release);
          }
        }
        if(next==std::numeric_limits<Clk_t>::max()&&!publications.empty()) {
          // No known request can progress. The external scheduler must
          // publish another event before the controller can advance.
          if(!unpublished)throw std::runtime_error("joint no progress after all external publications");
          co_yield 2;continue;
        }
        if(refresh&&cas_owner==std::numeric_limits<uint64_t>::max())next=std::min(next,next_ref);
        if(next==std::numeric_limits<Clk_t>::max())throw std::runtime_error("joint no progress");
        clk=std::min(horizon,jump?std::max(clk+1,next):clk+1);
      }
      if(clk>1000000000LL)throw std::runtime_error("joint cycle bound");
    }
  }
  void run(const Input* in,uint64_t count,uint64_t* completion,bool streams=false) {
    auto execution=execute(in,count,completion,streams);
    if(execution.advance()!=0)throw std::runtime_error("one-shot replay requires all external events");
  }
};
struct Session {
  std::span<const Input> inputs;
  Replay replay;
  Execution execution;
  bool started=false,watch_registered=false;
  Session(const char* config,const Input* in,uint64_t count,uint64_t* completion,
          bool refresh,bool jump,int slots,int capacity,int ranks)
      :inputs(in,count),replay(config,refresh,jump,slots,capacity,ranks),
       execution(replay.execute(inputs.data(),count,completion,true)) {
    replay.publications.assign(count,FenceRanges::unavailable);
    replay.unpublished=std::count_if(inputs.begin(),inputs.end(),[](const auto& input){return input.kind==10;});
  }
};
}
extern "C" const char* pim_joint_error(){return error.c_str();}
extern "C" uint64_t pim_wck_sync_issued(){return sync_issued;}
extern "C" uint64_t pim_wck_sync_reused(){return sync_reused;}
extern "C" void* pim_session_open(const char* config,const Input* in,uint64_t n,uint64_t* completion,
                                   int refresh,int jump,int slots,int capacity,int ranks) {
  try {
    if(!n)throw std::runtime_error("session needs at least one request");
    return new Session(config,in,n,completion,refresh,jump,slots,capacity,ranks);
  } catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" int pim_session_advance(void* handle,uint64_t horizon,Stats* stats,uint64_t* clock) {
  try {
    auto& session=*static_cast<Session*>(handle);
    auto& replay=session.replay;
    if(horizon>uint64_t(std::numeric_limits<Clk_t>::max())||horizon<uint64_t(replay.clk))
      throw std::runtime_error("session horizon must be monotonic and representable");
    replay.horizon=horizon;
    session.started=true;
    auto start=std::chrono::steady_clock::now();
    int reason=session.execution.advance();
    replay.stats.wall_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    *stats=replay.stats;*clock=replay.clk;
    return reason;
  } catch(const std::exception& e){error=e.what();return -1;}
}
namespace {
class ChannelWorkers {
  std::vector<void*> handles;
  std::vector<std::thread> threads;
  std::vector<std::string> failures;
  std::mutex mutex;
  std::condition_variable work, finished;
  uint64_t generation=0, horizon=0;
  size_t outstanding=0;
  std::atomic<size_t> next{0};
  bool stopping=false;
  Stats* stats=nullptr;
  uint64_t* clocks=nullptr;
  int* reasons=nullptr;

  void run() {
    uint64_t seen=0;
    std::unique_lock lock(mutex);
    while(true) {
      work.wait(lock,[&]{return stopping||generation!=seen;});
      if(stopping)return;
      seen=generation;
      lock.unlock();
      for(size_t channel=next.fetch_add(1);channel<handles.size();channel=next.fetch_add(1)) {
        reasons[channel]=pim_session_advance(handles[channel],horizon,stats+channel,clocks+channel);
        if(reasons[channel]<0)failures[channel]=error;
      }
      lock.lock();
      if(!--outstanding)finished.notify_one();
    }
  }
public:
  ChannelWorkers(void* const* values,size_t count,size_t workers)
      :handles(values,values+count),failures(count) {
    if(!count||!workers)throw std::runtime_error("channel workers require nonempty sessions");
    workers=std::min(workers,count);
    try {
      for(size_t i=0;i<workers;++i)threads.emplace_back([this]{run();});
    } catch(...) {close();throw;}
  }
  void advance(uint64_t limit,Stats* output_stats,uint64_t* output_clocks,int* output_reasons) {
    std::unique_lock lock(mutex);
    horizon=limit;stats=output_stats;clocks=output_clocks;reasons=output_reasons;
    outstanding=threads.size();next=0;++generation;work.notify_all();
    finished.wait(lock,[&]{return !outstanding;});
    for(size_t channel=0;channel<handles.size();++channel)
      if(reasons[channel]<0)throw std::runtime_error(failures[channel]);
  }
  void close() {
    {std::lock_guard lock(mutex);stopping=true;work.notify_all();}
    for(auto& thread:threads)if(thread.joinable())thread.join();
  }
  ~ChannelWorkers(){close();}
};
}
extern "C" void* pim_channels_open(void* const* sessions,uint64_t count,uint64_t workers) {
  try {return new ChannelWorkers(sessions,count,workers);}
  catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" int pim_channels_advance(void* handle,uint64_t horizon,Stats* stats,uint64_t* clocks,int* reasons) {
  try {static_cast<ChannelWorkers*>(handle)->advance(horizon,stats,clocks,reasons);return 0;}
  catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" void pim_channels_close(void* handle){delete static_cast<ChannelWorkers*>(handle);}
extern "C" int pim_session_publish(void* handle,uint64_t index,uint64_t timestamp) {
  try {
    auto& session=*static_cast<Session*>(handle);
    if(index>=session.inputs.size()||session.inputs[index].kind!=10)
      throw std::runtime_error("publication must reference an external request");
    auto& event=session.replay.publications[index];
    if(event!=FenceRanges::unavailable)throw std::runtime_error("external event already published");
    if(timestamp<uint64_t(session.replay.clk)||timestamp>uint64_t(std::numeric_limits<Clk_t>::max()))
      throw std::runtime_error("publication precedes committed controller time or overflows");
    event=timestamp;
    if(session.replay.actual_submissions)session.replay.published_inputs.push_back(index);
    --session.replay.unpublished;
    return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int pim_session_publish_many(void* handle,const uint64_t* indices,uint64_t count,uint64_t timestamp) {
  try {
    auto& session=*static_cast<Session*>(handle);
    if(timestamp<uint64_t(session.replay.clk)||timestamp>uint64_t(std::numeric_limits<Clk_t>::max()))
      throw std::runtime_error("publication precedes committed controller time or overflows");
    for(uint64_t i=0;i<count;++i) {
      auto index=indices[i];
      if(i&&index<=indices[i-1])throw std::runtime_error("publication batch must have increasing unique indices");
      if(index>=session.inputs.size()||session.inputs[index].kind!=10)
        throw std::runtime_error("publication must reference an external request");
      if(session.replay.publications[index]!=FenceRanges::unavailable)
        throw std::runtime_error("external event already published");
    }
    for(uint64_t i=0;i<count;++i) {
      session.replay.publications[indices[i]]=timestamp;
      if(session.replay.actual_submissions)session.replay.published_inputs.push_back(indices[i]);
    }
    session.replay.unpublished-=count;
    return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int pim_session_watch(void* handle,const uint64_t* offsets,uint64_t groups,const uint64_t* indices,uint64_t edges) {
  try {
    auto& session=*static_cast<Session*>(handle);auto& replay=session.replay;
    if(session.started||session.watch_registered)throw std::runtime_error("completion groups must be registered once before execution");
    if(groups>=None||edges>=None||offsets[0]!=0||offsets[groups]!=edges)
      throw std::runtime_error("invalid completion group bounds");
    for(uint64_t group=0;group<groups;++group)
      if(offsets[group]>offsets[group+1])throw std::runtime_error("completion group offsets must increase");
    for(uint64_t edge=0;edge<edges;++edge)
      if(indices[edge]>=session.inputs.size())throw std::runtime_error("completion group request is out of range");
    replay.watch_head.assign(session.inputs.size(),None);
    replay.watch_remaining.resize(groups);replay.watch_maximum.assign(groups,0);
    replay.watch_next.resize(edges);replay.watch_group.resize(edges);
    for(uint64_t group=0;group<groups;++group) {
      replay.watch_remaining[group]=offsets[group+1]-offsets[group];
      if(!replay.watch_remaining[group])replay.notifications.push_back({group,0});
      for(uint64_t edge=offsets[group];edge<offsets[group+1];++edge) {
        replay.watch_group[edge]=group;replay.watch_next[edge]=replay.watch_head[indices[edge]];
        replay.watch_head[indices[edge]]=edge;
      }
    }
    session.watch_registered=true;
    return 0;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" int64_t pim_session_observe(void* handle,uint64_t* output,uint64_t capacity) {
  try {
    auto& ready=static_cast<Session*>(handle)->replay.notifications;
    uint64_t count=ready.size();
    if(!capacity)return count;
    if(capacity<count)throw std::runtime_error("completion notification capacity is too small");
    for(uint64_t i=0;i<count;++i){output[2*i]=ready[i][0];output[2*i+1]=ready[i][1];}
    ready.clear();return count;
  } catch(const std::exception& e){error=e.what();return -1;}
}
extern "C" void pim_session_close(void* handle) {delete static_cast<Session*>(handle);}
extern "C" int pim_joint_replay(const char* config,const Input* in,uint64_t n,uint64_t* completion,Stats* stats,int refresh,int jump) {
  try {
    auto start=std::chrono::steady_clock::now();
    Replay replay(config,refresh,jump);replay.run(in,n,completion);
    replay.stats.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    *stats=replay.stats;return 0;
  } catch(const std::exception& e){error=e.what();return 1;}
}

extern "C" int pim_stream_replay(const char* config,const Input* in,uint64_t n,uint64_t* completion,Stats* stats,int refresh,int jump) {
  try {
    auto start=std::chrono::steady_clock::now();
    Replay replay(config,refresh,jump);replay.run(in,n,completion,true);
    replay.stats.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    *stats=replay.stats;return 0;
  } catch(const std::exception& e){error=e.what();return 1;}
}

extern "C" int pim_hcf_replay(const char* config,const Input* in,uint64_t n,uint64_t* completion,Stats* stats,int refresh,int jump,int streams,int slots,int capacity) {
  try {
    auto start=std::chrono::steady_clock::now();
    Replay replay(config,refresh,jump,slots,capacity);replay.run(in,n,completion,streams);
    replay.stats.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    *stats=replay.stats;return 0;
  } catch(const std::exception& e){error=e.what();return 1;}
}

extern "C" int pim_channel_replay(const char* config,const Input* in,uint64_t n,uint64_t* completion,Stats* stats,int refresh,int jump,int slots,int capacity) {
  try {
    auto start=std::chrono::steady_clock::now();
    Replay replay(config,refresh,jump,slots,capacity,2);replay.run(in,n,completion,true);
    replay.stats.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    *stats=replay.stats;return 0;
  } catch(const std::exception& e){error=e.what();return 1;}
}
