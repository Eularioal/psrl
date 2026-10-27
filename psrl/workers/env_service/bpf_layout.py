"""Resolve the small set of kernel field offsets used by GPU identity tracing."""

import struct
from pathlib import Path


class KernelTypes:
    """Read kernel BTF metadata without guessing offsets from a kernel version."""

    def __init__(self, path=Path("/sys/kernel/btf/vmlinux")):
        blob = path.read_bytes()
        magic, version, _, header, offset, length, strings, string_length = struct.unpack_from("<HBBIIIII", blob)
        if magic != 0xEB9F or version != 1:
            raise RuntimeError("GPU identity tracing requires little endian BTF version 1.")
        self.strings = blob[header + strings : header + strings + string_length]
        data = blob[header + offset : header + offset + length]
        self.types = [None]
        cursor = 0
        fixed = {1: 4, 2: 0, 3: 12, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 14: 4, 16: 0, 17: 4, 18: 0}
        variable = {4: 12, 5: 12, 6: 8, 13: 8, 15: 12, 19: 12}
        while cursor < len(data):
            name, info, value = struct.unpack_from("<III", data, cursor)
            kind, count = info >> 24 & 31, info & 65535
            cursor += 12
            if kind not in fixed and kind not in variable:
                raise RuntimeError(f"Unsupported kernel BTF kind {kind!r}.")
            size = fixed[kind] if kind in fixed else variable[kind] * count
            payload = data[cursor : cursor + size]
            if len(payload) != size:
                raise RuntimeError("Kernel BTF record is truncated.")
            self.types.append((self.string(name), kind, value, count, bool(info >> 31), payload))
            cursor += size

    def string(self, offset):
        return self.strings[offset : self.strings.index(0, offset)].decode()

    def named(self, name, kind):
        matches = [entry for entry in self.types[1:] if entry[0] == name and entry[1] == kind]
        layouts = set()
        for entry in matches:
            if kind == 4:
                members = [struct.unpack_from("<III", entry[5], index * 12) for index in range(entry[3])]
                shape = tuple((self.string(member), offset) for member, _, offset in members)
            else:
                shape = entry[5]
            layouts.add((entry[2], entry[3], entry[4], shape))
        if len(layouts) != 1:
            raise RuntimeError(f"Kernel BTF does not uniquely define {name!r}.")
        return matches[0]

    def member(self, structure, field):
        entry = self.named(structure, 4)
        for index in range(entry[3]):
            name, type_id, offset = struct.unpack_from("<III", entry[5], index * 12)
            if self.string(name) == field:
                if offset % 8 or entry[4] and offset >> 24:
                    raise RuntimeError(f"Unsupported bit field {structure!r}.{field!r}.")
                return offset // 8, type_id
        raise RuntimeError(f"Kernel BTF lacks {structure!r}.{field!r}.")

    def enum(self, name, field):
        entry = self.named(name, 6)
        for index in range(entry[3]):
            member, value = struct.unpack_from("<II", entry[5], index * 8)
            if self.string(member) == field:
                return value
        raise RuntimeError(f"Kernel BTF lacks enum value {field!r}.")

    def identity_defines(self):
        fields = {
            "TASK_PID": ("task_struct", "pid"),
            "TASK_TGID": ("task_struct", "tgid"),
            "TASK_LEADER": ("task_struct", "group_leader"),
            "TASK_BIRTH": ("task_struct", "start_time"),
            "TASK_CGROUPS": ("task_struct", "cgroups"),
            "CSS_SUBSYS": ("css_set", "subsys"),
            "CSS_CGROUP": ("cgroup_subsys_state", "cgroup"),
            "CSS_PARENT": ("cgroup_subsys_state", "parent"),
            "CGROUP_KN": ("cgroup", "kn"),
            "KN_ID": ("kernfs_node", "id"),
        }
        values = {macro: self.member(*field)[0] for macro, field in fields.items()}
        values["DEVICES_INDEX"] = self.enum("cgroup_subsys_id", "devices_cgrp_id")
        inode_type = self.types[self.member("kernfs_node", "id")[1]]
        values["KN_ID_LOW32"] = int(inode_type[1] == 5 and inode_type[0] == "kernfs_node_id")
        return values
