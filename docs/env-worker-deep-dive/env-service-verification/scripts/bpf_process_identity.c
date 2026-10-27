/* Observe one randomly named test process using a temporary raw tracepoint. */
#include <errno.h>
#include <linux/bpf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define I(c, d, s, o, v) ((struct bpf_insn){.code=(c), .dst_reg=(d), .src_reg=(s), .off=(o), .imm=(v)})

static int bpf_call(enum bpf_cmd command, union bpf_attr *attr) {
    return syscall(__NR_bpf, command, attr, sizeof(*attr));
}

int main(int argc, char **argv) {
    int map = -1, program = -1, attachment = -1, result = 3;
    const char *stage = "arguments";
    char verifier[65536] = {0}, tag[16] = {0};
    uint64_t first, second, value = 0;
    uint32_t key = 0;
    if (argc != 2 || strlen(argv[1]) != 15) {
        fprintf(stderr, "Expected one unique 15 character process name.\n");
        return 2;
    }
    memcpy(tag, argv[1], 15);
    memcpy(&first, tag, 8);
    memcpy(&second, tag + 8, 8);
    struct rlimit limit;
    if (getrlimit(RLIMIT_MEMLOCK, &limit) == 0 && limit.rlim_cur < 1048576) {
        limit.rlim_cur = limit.rlim_max < 1048576 ? limit.rlim_max : 1048576;
        if (setrlimit(RLIMIT_MEMLOCK, &limit) != 0)
            perror("setrlimit");
    }
    union bpf_attr attr = {0};
    attr.map_type = BPF_MAP_TYPE_ARRAY;
    attr.key_size = sizeof(key);
    attr.value_size = sizeof(value);
    attr.max_entries = 1;
    stage = "map_create";
    map = bpf_call(BPF_MAP_CREATE, &attr);
    if (map < 0) goto done;

    /* Reject every event except the exact random name assigned to our child. */
    struct bpf_insn instructions[] = {
        I(0xbf, 1, 10, 0, 0), I(0x07, 1, 0, 0, -16), I(0xb7, 2, 0, 0, 16),
        I(0x85, 0, 0, 0, BPF_FUNC_get_current_comm),
        I(0x79, 6, 10, -16, 0),
        I(0x18, 7, 0, 0, (uint32_t)first), I(0, 0, 0, 0, first >> 32),
        I(0x5d, 6, 7, 16, 0),
        I(0x79, 6, 10, -8, 0),
        I(0x18, 7, 0, 0, (uint32_t)second), I(0, 0, 0, 0, second >> 32),
        I(0x5d, 6, 7, 12, 0),
        I(0x85, 0, 0, 0, BPF_FUNC_get_current_pid_tgid),
        I(0x7b, 10, 0, -24, 0), I(0x62, 10, 0, -28, 0),
        I(0x18, 1, BPF_PSEUDO_MAP_FD, 0, map), I(0, 0, 0, 0, 0),
        I(0xbf, 2, 10, 0, 0), I(0x07, 2, 0, 0, -28),
        I(0xbf, 3, 10, 0, 0), I(0x07, 3, 0, 0, -24),
        I(0xb7, 4, 0, 0, BPF_ANY), I(0x85, 0, 0, 0, BPF_FUNC_map_update_elem),
        I(0xb7, 0, 0, 0, 0), I(0x95, 0, 0, 0, 0),
    };
    /* Both rejection branches return zero without changing a process or its syscall. */
    instructions[7].off = 23 - 7 - 1;
    instructions[11].off = 23 - 11 - 1;
    memset(&attr, 0, sizeof(attr));
    attr.prog_type = BPF_PROG_TYPE_RAW_TRACEPOINT;
    attr.insn_cnt = sizeof(instructions) / sizeof(instructions[0]);
    attr.insns = (uint64_t)(uintptr_t)instructions;
    attr.license = (uint64_t)(uintptr_t)"GPL";
    attr.log_level = 1;
    attr.log_size = sizeof(verifier);
    attr.log_buf = (uint64_t)(uintptr_t)verifier;
    stage = "program_load";
    program = bpf_call(BPF_PROG_LOAD, &attr);
    if (program < 0) goto done;
    memset(&attr, 0, sizeof(attr));
    attr.raw_tracepoint.name = (uint64_t)(uintptr_t)"sys_enter";
    attr.raw_tracepoint.prog_fd = program;
    stage = "raw_tracepoint_attach";
    attachment = bpf_call(BPF_RAW_TRACEPOINT_OPEN, &attr);
    if (attachment < 0) goto done;
    puts("READY");
    fflush(stdout);
    stage = "map_lookup";
    for (int attempt = 0; attempt < 400; attempt++) {
        memset(&attr, 0, sizeof(attr));
        attr.map_fd = map;
        attr.key = (uint64_t)(uintptr_t)&key;
        attr.value = (uint64_t)(uintptr_t)&value;
        if (bpf_call(BPF_MAP_LOOKUP_ELEM, &attr) < 0) goto done;
        if (value) {
            printf("{\"host_tgid\":%u,\"host_tid\":%u}\n", (unsigned)(value >> 32), (unsigned)value);
            result = 0;
            goto done;
        }
        struct timespec delay = {.tv_nsec = 50000000};
        nanosleep(&delay, NULL);
    }
    stage = "event_timeout";
    errno = ETIMEDOUT;
done:
    if (result != 0) {
        fprintf(stderr, "%s: errno=%d %s\n", stage, errno, strerror(errno));
        if (verifier[0]) fprintf(stderr, "%s\n", verifier);
    }
    if (attachment >= 0) close(attachment);
    if (program >= 0) close(program);
    if (map >= 0) close(map);
    return result;
}
