/* Track registered device cgroups, process birth identities and invalidation events. */
#include "identity_offsets.h"

typedef unsigned int u32;
typedef unsigned long long u64;
#define SEC(name) __attribute__((section(name), used))
#define INLINE static __attribute__((always_inline)) inline
#define UINT(name, value) int (*name)[value]
#define TYPE(name, value) value *name

struct identity { u64 birth_ns; u64 cgroup; u64 cookie; };
struct raw_args { u64 args[4]; };
struct { UINT(type, 1); UINT(max_entries, 256); TYPE(key, u64); TYPE(value, u64); } registrations SEC(".maps");
struct { UINT(type, 1); UINT(max_entries, 8192); TYPE(key, u32); TYPE(value, struct identity); } identities SEC(".maps");
struct { UINT(type, 2); UINT(max_entries, 1); TYPE(key, u32); TYPE(value, u64); } health SEC(".maps");

static void *(*lookup)(void *, const void *) = (void *)1;
static long (*update)(void *, const void *, const void *, u64) = (void *)2;
static long (*remove_key)(void *, const void *) = (void *)3;
static long (*read_kernel)(void *, u32, const void *) = (void *)4;
static void *(*current_task)(void) = (void *)35;

INLINE void fault(void) {
    u32 zero = 0;
    u64 *errors = lookup(&health, &zero);
    if (errors) __sync_fetch_and_add(errors, 1);
}

#define READ(value, base, offset) read_kernel(&(value), sizeof(value), (char *)(base) + (offset))

INLINE void *device_css(void *task) {
    void *set = 0, *css = 0;
    if (READ(set, task, TASK_CGROUPS) || !set) return 0;
    if (READ(css, set, CSS_SUBSYS + DEVICES_INDEX * 8)) return 0;
    return css;
}

INLINE u64 owner(void *css, u64 *matched) {
#pragma unroll
    for (int depth = 0; depth < 16; depth++) {
        if (!css) return 0;
        void *cgroup = 0, *node = 0;
        u64 id = 0;
        if (READ(cgroup, css, CSS_CGROUP) || !cgroup || READ(node, cgroup, CGROUP_KN) || !node)
            return 0;
        if (READ(id, node, KN_ID)) return 0;
#if KN_ID_LOW32
        id = (u32)id;
#endif
        u64 *cookie = lookup(&registrations, &id);
        if (cookie) { *matched = id; return *cookie; }
        if (READ(css, css, CSS_PARENT)) return 0;
    }
    return 0;
}

INLINE int capture(void *task) {
    void *leader = 0;
    u32 pid = 0;
    struct identity value = {0};
    if (READ(pid, task, TASK_TGID) || !pid || READ(leader, task, TASK_LEADER) || !leader) return 0;
    value.cookie = owner(device_css(task), &value.cgroup);
    if (!value.cookie) { remove_key(&identities, &pid); return 0; }
    if (leader != task) {
        u64 group = 0;
        if (owner(device_css(leader), &group) != value.cookie || group != value.cgroup) {
            remove_key(&identities, &pid);
            return 0;
        }
    }
    if (READ(value.birth_ns, leader, TASK_BIRTH) || !value.birth_ns) { fault(); return 0; }
    if (update(&identities, &pid, &value, 0)) fault();
    return 0;
}

SEC("raw_tracepoint/sys_enter")
int on_syscall(struct raw_args *context) {
    /* CUDA enters the driver through ioctl, while getpid supports the startup handshake. */
    if (context->args[1] != IOCTL_NR && context->args[1] != GETPID_NR) return 0;
    return capture(current_task());
}

SEC("raw_tracepoint/sched_process_fork")
int on_fork(struct raw_args *context) { return capture((void *)context->args[1]); }

SEC("raw_tracepoint/sched_process_exec")
int on_exec(struct raw_args *context) {
    u32 old_pid = context->args[1];
    remove_key(&identities, &old_pid);
    return capture((void *)context->args[0]);
}

INLINE int invalidate(void *task) {
    u32 pid = 0;
    if (!READ(pid, task, TASK_TGID)) remove_key(&identities, &pid);
    return 0;
}

SEC("raw_tracepoint/sched_process_exit")
int on_exit(struct raw_args *context) { return invalidate((void *)context->args[0]); }

SEC("raw_tracepoint/cgroup_attach_task")
int on_move(struct raw_args *context) { return invalidate((void *)context->args[2]); }

SEC("raw_tracepoint/cgroup_transfer_tasks")
int on_transfer(struct raw_args *context) { return invalidate((void *)context->args[2]); }

char program_license[] SEC("license") = "GPL";
