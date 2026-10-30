# DEVSIM MCP

这是一个单文件 Python MCP 服务，通过系统环境中安装的 `devsim` 驱动 TCAD 仿真。服务不依赖本目录中的 DEVSIM 源码，也不包含绘图功能；仿真数据会保存为可下载的结果文件。

## 安装与启动

```bash
pip install devsim mcp
python devsim_mcp.py --data-dir ./devsim-results
```

默认使用 MCP stdio。也支持：

```bash
python devsim_mcp.py --transport sse --host 127.0.0.1 --port 8000
python devsim_mcp.py --transport streamable-http --host 127.0.0.1 --port 8000
```

## 工具

- **会话与运行时**：创建、查询、重置、关闭独立仿真会话，查看 DEVSIM 版本和实际加载路径。
- **网格与设备**：定义 1D/2D 结构化网格，导入 Gmsh 网格，完成网格、设备和拓扑管理。
- **物理模型**：设置参数和材料，创建 solution、node/edge/element/contact/interface model 与 equation，提供 Silicon/Oxide physics preset。
- **仿真任务**：执行 DC、AC、noise、transient 求解，以及 bias、frequency、transient sweep；支持后台任务进度、等待和取消。
- **查询与诊断**：读取字段、模型、方程、接触电流/电荷、电路状态、矩阵、设备拓扑，并支持 JSON-compatible 原生 DEVSIM 调用和批量调用。
- **结果与恢复**：保存 checkpoint，导出 DEVSIM、VTK、Tecplot 文件，保存 sweep JSON 和大数组 NPZ，使用 artifact manifest 查询结果。
- **脚本导出**：将已记录的 JSON-compatible 操作导出为可独立运行的 Python 脚本。

长任务工具返回 `job_id`，通过 `job_status` 或 `job_wait` 获取进度和最终状态。每个 session 使用独立 worker，session 关闭或服务退出时会回收 worker 进程。
