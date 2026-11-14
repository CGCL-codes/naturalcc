# Zephyr v6 编译上下文复现

此修订修复 `kernel/atomic_c.c` 的原子操作重定义，以及 `kernel/banner.c` 的 CPU 配置和 syscall 头文件缺失。编译上下文只增加数据库、配置和头文件；另外，independent 与 independent-hard 各在原有 z_impl_device_init 的一个空指针位置引入字面量 strspn 调用结果推理，并同步更新 GT，每套仍为 60 缺陷 + 60 对照。该调用返回零，已用独立 sanitizer 见证验证实际空指针访问，记录见 ../literal-span-witness.json。完整参考规则实现该语义，学生 E-off 配置保留推理缺口；原始引擎诊断不受过滤。

`overlays/zephyr/` 中的头文件由 Zephyr v3.7.0 固定提交 `36940db938a8f4a1e919496793ed439850a221c2` 的真实 CMake/Kconfig/syscall/devicetree 流程生成，使用主机板 `native_posix/native/64`。不是手写的 autoconf 或空类型桩。原生配置和 C 原子实现配置分别保存在 `build_context/native64/`、`build_context/atomic-c/`，避免混用互斥实现。

编译数据库中的 206 条记录包含两种来源，逐文件记录在 `build_context/provenance.json`：`cmake-real-entry` 是实际 CMake 条目；`derived-source-audit-profile` 借用主机分析参数，不表示该文件被实际板级构建选中。参数均为工程内相对路径，搬移工程后可自动读取；`macros: 4` 只统计直接 `-D` 参数，实际 CONFIG 来自 `-imacros autoconf.h`。

**Clang 仍为部分覆盖。** 独立 syntax-only 复测四套通过 113/206、114/206、113/206、114/206 个文件。剩余文件需要其他真实功能、板级配置或外部依赖。完整诊断见 `../zephyr-clang-syntax-audit.json`；不能用 120/120 标注操作可计分代替所有 C 文件 Clang 成功。

复现需要 CMake、Ninja、主机 GCC 和独立 Python 环境。实际使用 Python 3.12、kconfiglib 14.1.0 及该提交内的 python-devicetree；同时安装 pyelftools、PyYAML、packaging、pykwalify、west、pyserial。将 `ZEPHYR_BASE` 指向上述提交的完整源码，`BUILD_PYTHON` 指向该环境解释器，`CONTEXT_APP` 指向本目录 `atomic-c`：

```bash
cmake -GNinja -S "$ZEPHYR_BASE/samples/hello_world" -B /tmp/c3-zephyr-native \
  -DBOARD=native_posix/native/64 -DZEPHYR_TOOLCHAIN_VARIANT=host \
  -DPython3_EXECUTABLE="$BUILD_PYTHON" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/c3-zephyr-native --target zephyr_generated_headers -j2
cmake -GNinja -S "$CONTEXT_APP" -B /tmp/c3-zephyr-atomic \
  -DBOARD=native_posix/native/64 -DZEPHYR_TOOLCHAIN_VARIANT=host \
  -DPython3_EXECUTABLE="$BUILD_PYTHON" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build /tmp/c3-zephyr-atomic --target zephyr_generated_headers -j2
python scripts/export_contract3_zephyr_context.py --source "$ZEPHYR_BASE" \
  --native-build /tmp/c3-zephyr-native --atomic-build /tmp/c3-zephyr-atomic
python scripts/build_contract3_engineering_v6.py --project zephyr
```

生成步骤需要完整上游源码；交付快照已经含有导出的头文件，普通静态检测无需再次运行 CMake。主机 C 原子配置的 pristine hello_world 已完整编译链接成功，未执行目标程序。该结果不代表被注入缺陷的全部工程可完整构建。
