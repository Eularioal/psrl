# Linux教学：CPU演示脚本与原始证据

[上级导航](../README.md) · [总导航](../../README.md)

完整知识讲解集中在[系统基础](../系统基础/README.md)中的进程与文件映射、inode与链接两篇。本目录保留它们引用的CPU演示脚本与原始结果，运行方法和条件见对应正文。

<!-- figure:file-references -->
这张图沿同一个场景解释：文件名删了，为什么进程还能读？空间什么时候才可回收？

[![文件名、inode与打开引用：名字消失后谁还在用文件（1/2）](../../_static/img/document-figures/file-references-screen.png)](../../_static/img/document-figures/file-references.png)

[![文件名、inode与打开引用：名字消失后谁还在用文件（2/2）](../../_static/img/document-figures/file-references-screen-02.png)](../../_static/img/document-figures/file-references.png)

同一文件对象 I 有两个名字 a、b，另有进程通过 fd 打开它。 名字是查找入口，打开引用是进程的使用权。删除入口不等于立即销毁正在使用的对象。 省略文件系统内部细节。I、J 是教学标记，删除与截断文件的后果不同。 [查看矢量图](../../_static/img/document-figures/file-references.svg)。
<!-- /figure:file-references -->

## 文件与子目录

- [inode-demo-results.json](inode-demo-results.json)
- [inode_demo.py](inode_demo.py)
- [mmap-sigbus-demo.py](mmap-sigbus-demo.py)
