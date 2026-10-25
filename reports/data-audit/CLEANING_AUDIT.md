# 只读数据清洗审计：实际副本独立复核
## 1. 范围、方法与错误
- 唯一项目范围：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516`；只读取被允许的 dataset.yaml、reports/data_manifest.json、reports/DATA_DISTRIBUTION.md 和副本 data/train、data/val。未读 README（无必要），未读取测试数据、凭据、项目日志、其他历史产物、主目录或其他 worktree 内容。初始一次范围内 Glob 列出了根范围文件名元数据（含非目标产物名称），未打开这些非目标文件；后续均定向读取。
- 初始只读子agent仅向会话scratch写脚本和结果；主agent将工具与证据归档到reports/data-audit。归档版仅刷新该目录内的JSON及报告，不修改数据、标签、划分、配置或训练入口；本轮未训练、安装、联网或启动3LC服务。
- Python 仅 E:\anaconda\envs\dl-3lc\python.exe，3.12.15 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:14:50) [MSC v.1942 64 bit (AMD64)]；Pillow 12.3.0，numpy 2.5.2；使用 -B 禁止 bytecode 写入。
- 从 manifest 的 train/val images 条目先按 SHA256 分组，独立使用当前磁盘全部图片字节重新分组并逐项比对，恰好一致的16组/16对；所有16对另作直接 bytes== 验证。
- 共读取并重新核验 17668/17668 个图片及标签文件的字节数与完整 SHA256；实际文件集合与 manifest 一致。7852 train + 982 val 图片，框数 72273/11615。无缺失/孤儿标签、命名解析失败、哈希/字节数不符。
- 16对的32张图片及2张空标签图片均 Pillow Image.open -> load 完整解码，numpy 比较原生像素与 RGB 像素；全部16对尺寸640×640、RGB、JPEG且像素逐元素一致。其他图片仅读取字节/hash及文件名，不把本次称为所有图片解码检查。
- 标签用原始字节/完整文本对比；非空行解析为 class + Decimal(cx,cy,w,h)。精确行顺序比较、精确无序多重集合比较；再按类做最小总 L∞ 坐标差矩形指派（自写 Hungarian，虚拟项承接多余框）。另做不按类别的几何指派，仅用于提示相近坐标/不同类别的条目，不能据此断言同一物体。
- 16对的 val 坐标字段全部为6位小数；共同框逐字段绝对差<=0.0000005 为六位小数舍入相容（含半舍入方向差异），对应640输入参数差<=0.00032 px；不是精确标签相同。匹配之外的增减框、类别/几何变化不能用舍入解释。
- 初始子agent统计与标签比较退出码均0、计算errors=[]。其报告生成最初因普通字符串反斜杠触发SyntaxWarning，修为raw字符串后退出码0；归档版执行情况见SUMMARY.md及run_*.log。第2步补齐所有16组矩形/跨类几何比对。
- 复现顺序（仅重建本目录审计派生结果；不修改数据）：
```powershell
$py='E:\anaconda\envs\dl-3lc\python.exe'
$s='E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\reports\data-audit'
& $py -B "$s\cleaning_audit.py"
& $py -B "$s\cleaning_audit_labels.py"
& $py -B "$s\cleaning_audit_report.py"
```
## 2. 16对逐组文件与标签证据
- 类别映射来自 dataset.yaml:10–15：0=truck，1=car，2=van，3=bus。以下T/V为train/val；所有组去 .rf 散列后的 MVI/帧标识两侧一致，文件名不同。
- **所有16对：图片完整字节/SHA256/尺寸/原生与RGB像素相同；标签原始字节和精确文本均不相同；精确有序/无序数值标签也均不相同。4对仅舍入相容；12对框数或类别计数有结构差异。**
### 组01 MVI_40751_img00584
- T图片：`train/images/MVI_40751_img00584_jpg.rf.f236753cee927d4714e9914f764a6689.jpg`
- V图片：`val/images/MVI_40751_img00584_jpg.rf.d95818832c8fecdcf873d60681ab364e.jpg`
- 两文件各 55537 bytes，640×640 RGB/JPEG；SHA256=`5ca8d4da07e4ccbbb174d4960bba4794ec6a7cde40c7e32bee11249ba1248ee0`；直接字节及完整解码像素相同。
- T/V框数：6/5；T类别计数 `{'1': 4, '3': 2}`，V `{'1': 3, '3': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 256 bytes，SHA256=`0359ee4c46b177d0c20eb1732be0f67f4d64ddecf112e8c85b08b16a42446055`；V标签 193 bytes，SHA256=`0f60f9f2652fe593bb8f7db0e9a1bfeeadecfa42d10eae506d159709f6f4e9a3`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行4：class=1 coords=(0.22734375 0.55859375 0.071875 0.08671875)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组02 MVI_40751_img00601
- T图片：`train/images/MVI_40751_img00601_jpg.rf.42323e39ee41ec1fbf2b5d8e34f18607.jpg`
- V图片：`val/images/MVI_40751_img00601_jpg.rf.5f45a77f3048eef641e951a0363b778d.jpg`
- 两文件各 56321 bytes，640×640 RGB/JPEG；SHA256=`9fab984aa3885e49f1d152ba19fce5ca24fe44e9fe74606bc287624857be0555`；直接字节及完整解码像素相同。
- T/V框数：7/6；T类别计数 `{'1': 5, '3': 2}`，V `{'1': 4, '3': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 295 bytes，SHA256=`a9dd1a8aa7c6d77c8f8547fe9b37256953c77b50fc6e00a21deb5b572b9cc148`；V标签 232 bytes，SHA256=`156d020ce22fb97009a72d61afe122fe46191a73bbd821407f78defd5397c92c`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行4：class=1 coords=(0.225 0.56171875 0.071875 0.08515625)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组03 MVI_40751_img00650
- T图片：`train/images/MVI_40751_img00650_jpg.rf.8fd574d00c05c8376c68233ea8cc1e1e.jpg`
- V图片：`val/images/MVI_40751_img00650_jpg.rf.fbd17823b375c5676dec2626e5382891.jpg`
- 两文件各 55487 bytes，640×640 RGB/JPEG；SHA256=`1d9d8e5d0cfaaa57a6166c0cf5a60856f6ac475fccd7fbc12c1f82af1748b1a3`；直接字节及完整解码像素相同。
- T/V框数：6/5；T类别计数 `{'1': 4, '3': 2}`，V `{'1': 3, '3': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 250 bytes，SHA256=`39f120cfcb05b2ac9c4db833c7276f8b4eacbe1716817e09e2fff97a865cfe8d`；V标签 193 bytes，SHA256=`4ad00852a39f80b74005b1b4c299283a62f1132c16d5cacfb8350c17bb25b25e`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行4：class=1 coords=(0.225 0.56171875 0.071875 0.08515625)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组04 MVI_40751_img00811
- T图片：`train/images/MVI_40751_img00811_jpg.rf.5c2808e8181c99a4827bb73006aabf2c.jpg`
- V图片：`val/images/MVI_40751_img00811_jpg.rf.0731ce71b57c25a3a3b028f75b823a90.jpg`
- 两文件各 54936 bytes，640×640 RGB/JPEG；SHA256=`ece247d796d33b93a3db85e08f7c5dea78feb82759b7584544a955f491db11c7`；直接字节及完整解码像素相同。
- T/V框数：6/5；T类别计数 `{'1': 4, '3': 2}`，V `{'1': 3, '3': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 248 bytes，SHA256=`c778b27370e2dfc70a076dbbab72b9287996482eb8490e7b440644770c883fc4`；V标签 193 bytes，SHA256=`0fd1afc9049bdd54022513a929da7be26dbc3c620163c77e1ffb9f0debf7e131`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行4：class=1 coords=(0.2453125 0.5765625 0.075 0.07421875)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组05 MVI_40751_img00821
- T图片：`train/images/MVI_40751_img00821_jpg.rf.ecbfdb0cd3a996d8e9b4ad26562240ad.jpg`
- V图片：`val/images/MVI_40751_img00821_jpg.rf.5a2d3581ec50fdbdb7ccbcbf2526f479.jpg`
- 两文件各 54930 bytes，640×640 RGB/JPEG；SHA256=`94f9c20cc1dcb87e0bff8d86e476d67e7100ccdeec37588abe9a603f92f4ced7`；直接字节及完整解码像素相同。
- T/V框数：6/5；T类别计数 `{'1': 4, '3': 2}`，V `{'1': 3, '3': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 252 bytes，SHA256=`678fe39b27b896767a26390f6a37c7828f55b6d9a07ef488dc6620fd0aced440`；V标签 193 bytes，SHA256=`085f176efdd49cf6cd27659bd6f8f66563c13055255a835df63ea39a25f7b967`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行4：class=1 coords=(0.24375 0.59140625 0.08125 0.06875)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组06 MVI_40752_img00364
- T图片：`train/images/MVI_40752_img00364_jpg.rf.19508a6d3cc69d09dac5aa832ca5b802.jpg`
- V图片：`val/images/MVI_40752_img00364_jpg.rf.47b5b10af651bbb95991e49dc31c992d.jpg`
- 两文件各 54812 bytes，640×640 RGB/JPEG；SHA256=`5a027e34b62703293a2d7fa999850d126de2a6a40a67ebc8b76f3bda76e9ee4e`；直接字节及完整解码像素相同。
- T/V框数：9/10；T类别计数 `{'1': 7, '2': 2}`，V `{'1': 8, '2': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 376 bytes，SHA256=`23d27322a7f8f8af99e86e9a6d010cc3ad372c9745517695d946bd60eed75869`；V标签 388 bytes，SHA256=`9248cdeb33771891b65a99f1b76ce4de35cb47f33a7f914dd61462ec9ab4e91d`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=1 coords=(0.724609 0.101270 0.044531 0.047852)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组07 MVI_40752_img00387
- T图片：`train/images/MVI_40752_img00387_jpg.rf.9867061bfbb9a8a8017d6c68df5b57d4.jpg`
- V图片：`val/images/MVI_40752_img00387_jpg.rf.df9ede958c2a9b19a8481b8d6e7bea53.jpg`
- 两文件各 55354 bytes，640×640 RGB/JPEG；SHA256=`96f31e207d47314eba1e0594871c039631e36af65c95481aad0e2e1f249f80c4`；直接字节及完整解码像素相同。
- T/V框数：11/12；T类别计数 `{'1': 10, '2': 1}`，V `{'1': 11, '2': 1}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 437 bytes，SHA256=`248645680f9e5fef7afd2eaed7a506a6913228a86e66ada034fd0c6907f07220`；V标签 466 bytes，SHA256=`31a6cf363f3876c3eb43245c36d64f2d955eaaed048ef13550ae9c05996beb03`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=1 coords=(0.962500 0.159863 0.071875 0.078711)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组08 MVI_40752_img00391
- T图片：`train/images/MVI_40752_img00391_jpg.rf.e0ccab5b4ac4dfa32068a3793c57e3ab.jpg`
- V图片：`val/images/MVI_40752_img00391_jpg.rf.a54c86446e00c72319cd9ce1812912f4.jpg`
- 两文件各 55735 bytes，640×640 RGB/JPEG；SHA256=`48cb95c18c1274d8e709f4309b5cd7a6c15cb07b684f6f90eabc4d5c85912274`；直接字节及完整解码像素相同。
- T/V框数：10/11；T类别计数 `{'1': 9, '2': 1}`，V `{'1': 10, '2': 1}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 407 bytes，SHA256=`d0442660b49d1f8319c633054cd94d0a119ea074a13822f9b8fbbac2dd6e561c`；V标签 427 bytes，SHA256=`aa2f30e18f056e4e34ff191a6b44de27282c5d1829f5bd2d4a042c46d1d78029`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=1 coords=(0.968750 0.148340 0.059375 0.079492)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组09 MVI_40752_img00456
- T图片：`train/images/MVI_40752_img00456_jpg.rf.27e2d271e22445e666c6ade898cadde2.jpg`
- V图片：`val/images/MVI_40752_img00456_jpg.rf.a7c36c4dc5d59f5173fa6ece863e41c7.jpg`
- 两文件各 54274 bytes，640×640 RGB/JPEG；SHA256=`e3c28c6fdda7708543ee569644681007e386cc6405eea970727d98eabe5c01e8`；直接字节及完整解码像素相同。
- T/V框数：10/10；T类别计数 `{'1': 10}`，V `{'1': 10}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 426 bytes，SHA256=`9b596fc3b301aef6e7ce2b8b69724b39f5345dd2579df327e69c732658c17d3d`；V标签 388 bytes，SHA256=`42a5753c10056dba495989a27d7248037c85d49a0088e081879875715a339984`。
- 仅六位小数舍入相容；同类别匹配全部保留行序，最大归一化差 5E-7（×640=0.00032 px）；没有额外/缺少框。精确文本、精确数值有序和无序均不相同。

### 组10 MVI_40752_img00698
- T图片：`train/images/MVI_40752_img00698_jpg.rf.03515b303803084ac0223dac7c64f788.jpg`
- V图片：`val/images/MVI_40752_img00698_jpg.rf.7b8aefabff1ab660dc4075cc2f669c9c.jpg`
- 两文件各 51758 bytes，640×640 RGB/JPEG；SHA256=`225126d7d044a3d6726891026dc94d1bfeda76c2ba0f22fd84417783fd1ad08a`；直接字节及完整解码像素相同。
- T/V框数：6/8；T类别计数 `{'0': 1, '1': 5}`，V `{'0': 1, '1': 5, '2': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 261 bytes，SHA256=`b3d7117bba83943179643dce8d3a8d3b22faba99ea79891e24bbcb42714a65a5`；V标签 310 bytes，SHA256=`070d3413ec67d1a02b5b874f31e27e608a366a976924d9dce28ce66bfd3ea08f`。
- 类别约束矩形匹配最大差 0.068066；超过六位小数舍入的匹配框 1；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=2 coords=(0.973437 0.111914 0.053125 0.092578)
  - V独有行3：class=2 coords=(0.659375 0.151465 0.051562 0.057617)
  - 同类最小代价关联 T:5→V:2：T=(0.66015625 0.153125 0.05625 0.0609375)，V=(0.716406 0.085059 0.042188 0.042773)；|Δ(cx,cy,w,h)|=(0.05624975 0.068066 0.014062 0.0181645)；这是算法关联，不保证同一物体。
- **不按类几何关联** T行5 class1→V行3 class2：T=(0.66015625 0.153125 0.05625 0.0609375)，V=(0.659375 0.151465 0.051562 0.057617)；|Δ|=(0.00078125 0.001660 0.004688 0.0033205)；最大差×640=3.00032px。明显超过舍入，但不作为语义对应或标签真伪结论。

### 组11 MVI_40752_img00903
- T图片：`train/images/MVI_40752_img00903_jpg.rf.643b9e93cd9cb99673eaff691c1c6287.jpg`
- V图片：`val/images/MVI_40752_img00903_jpg.rf.03eda609c8c3a35f5a79a226660b4cda.jpg`
- 两文件各 54534 bytes，640×640 RGB/JPEG；SHA256=`478b6411fbe8d8a1f03ba2c304174c53cbcedb785390126756738a510beb6acb`；直接字节及完整解码像素相同。
- T/V框数：7/7；T类别计数 `{'1': 7}`，V `{'1': 7}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 294 bytes，SHA256=`07b697bfa99cc1ead8f2f105b14f1b51ff07b11f01758b353f82cfce77b466a7`；V标签 271 bytes，SHA256=`128413df6ddddddb91dc4120bb602f3d6afb524d3014998f96ea76eda03a7dff`。
- 仅六位小数舍入相容；同类别匹配全部保留行序，最大归一化差 5E-7（×640=0.00032 px）；没有额外/缺少框。精确文本、精确数值有序和无序均不相同。

### 组12 MVI_40752_img00912
- T图片：`train/images/MVI_40752_img00912_jpg.rf.92e9e14759dc1eb7220747d32e9099ff.jpg`
- V图片：`val/images/MVI_40752_img00912_jpg.rf.d1b8784ec508dda12367426cc5e7024a.jpg`
- 两文件各 54955 bytes，640×640 RGB/JPEG；SHA256=`d7944f6d87d8d60f848a56573724e9c18c78d88c951e9cbac703521ef7d75663`；直接字节及完整解码像素相同。
- T/V框数：7/7；T类别计数 `{'1': 7}`，V `{'1': 7}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 285 bytes，SHA256=`aabe9324b94db0a2149f68c160742fdced5b50c3bde59668d80872c08e1eff08`；V标签 271 bytes，SHA256=`31a4605cf1c19dad0fa86fe317c25f7e8367a071565275c93e3a1dce53c8f2bc`。
- 仅六位小数舍入相容；同类别匹配全部保留行序，最大归一化差 5E-7（×640=0.00032 px）；没有额外/缺少框。精确文本、精确数值有序和无序均不相同。

### 组13 MVI_40752_img01109
- T图片：`train/images/MVI_40752_img01109_jpg.rf.c5582d5f5f556f1540b1aae5c44e6683.jpg`
- V图片：`val/images/MVI_40752_img01109_jpg.rf.566cbec4e35efd505e5ad0ec8893bbce.jpg`
- 两文件各 52511 bytes，640×640 RGB/JPEG；SHA256=`68144624e8b70e69ebe0e0c56a89083f3691e33a404490d0c767e6b73c49dde8`；直接字节及完整解码像素相同。
- T/V框数：8/8；T类别计数 `{'1': 6, '2': 2}`，V `{'1': 6, '2': 2}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 338 bytes，SHA256=`c744893acbc486111e5cc1b4965dc428f7724ea2c0773489c1a77d9785f94216`；V标签 310 bytes，SHA256=`7e834470620a12e9ef5de1450bbb1bd6f4808d8beb08453aaa0dd4a9a64bb426`。
- 仅六位小数舍入相容；同类别匹配全部保留行序，最大归一化差 5E-7（×640=0.00032 px）；没有额外/缺少框。精确文本、精确数值有序和无序均不相同。

### 组14 MVI_40752_img01217
- T图片：`train/images/MVI_40752_img01217_jpg.rf.df9fe2dca3c07bc4341e7483da294571.jpg`
- V图片：`val/images/MVI_40752_img01217_jpg.rf.5afc7f252217fa7ed46c233c701ede3a.jpg`
- 两文件各 56410 bytes，640×640 RGB/JPEG；SHA256=`7e325e0921d936fc3062e48c52b33818994b862bd1e5fc2aad02ad7e280029ca`；直接字节及完整解码像素相同。
- T/V框数：7/9；T类别计数 `{'1': 6, '2': 1}`，V `{'1': 8, '2': 1}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 290 bytes，SHA256=`e71fd0f0d9f1eded1f6e498afe745aa15a592228667147bd0be298b0caf9c70d`；V标签 349 bytes，SHA256=`e3762484b74d58c2e7b2f20eb9d5c53e669e269824b0efb022a7858eb7439ef2`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=1 coords=(0.697266 0.088770 0.047656 0.045313)
  - V独有行2：class=1 coords=(0.793750 0.140820 0.043750 0.050000)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组15 MVI_40752_img01286
- T图片：`train/images/MVI_40752_img01286_jpg.rf.484a98a53057fc841f513237cc4c9bf8.jpg`
- V图片：`val/images/MVI_40752_img01286_jpg.rf.8240075d16fceb53e709d1953d978763.jpg`
- 两文件各 53502 bytes，640×640 RGB/JPEG；SHA256=`d22374a8ab3c1e893b463b8731b7fef6e2fcc88fffce5f8ae99f7a86820f0578`；直接字节及完整解码像素相同。
- T/V框数：8/10；T类别计数 `{'1': 7, '2': 1}`，V `{'0': 1, '1': 9}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 342 bytes，SHA256=`5304e583683fa8ca352144bf36c12a153a286708c3b2bfa393df59213cbe2137`；V标签 388 bytes，SHA256=`acf27c4ea5f7a996a42a39fa8b53a6ed9b34de1215acfc5ef844c7c5fe091812`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - T独有行8：class=2 coords=(0.58359375 0.13671875 0.05625 0.05703125)
  - V独有行1：class=1 coords=(0.584375 0.131445 0.056250 0.062500)
  - V独有行2：class=1 coords=(0.728125 0.107227 0.053906 0.057422)
  - V独有行10：class=0 coords=(0.725565 0.068594 0.066484 0.086632)
- **不按类几何关联** T行8 class2→V行1 class1：T=(0.58359375 0.13671875 0.05625 0.05703125)，V=(0.584375 0.131445 0.056250 0.062500)；|Δ|=(0.00078125 0.00527375 0.000000 0.00546875)；最大差×640=3.5px。明显超过舍入，但不作为语义对应或标签真伪结论。
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

### 组16 MVI_40752_img01434
- T图片：`train/images/MVI_40752_img01434_jpg.rf.810d34756130819013db8865be6d1e30.jpg`
- V图片：`val/images/MVI_40752_img01434_jpg.rf.cf39651eff877574dbc55982e24d7183.jpg`
- 两文件各 54754 bytes，640×640 RGB/JPEG；SHA256=`1818a2829007defbec8aaff8ca129ecf808945a91576426f1b244dc41e4b8e67`；直接字节及完整解码像素相同。
- T/V框数：8/10；T类别计数 `{'1': 8}`，V `{'1': 10}`。
- 标签路径：同图片stem，images→labels，后缀→.txt；T标签 345 bytes，SHA256=`9bcc36ea3b8519cdf7cf2efd60ae7d1b8fc730124b30d064548d0541e107250e`；V标签 388 bytes，SHA256=`941df0ebf5842166df16e9d81ec1e78b4d842330d395431c73caf7fb457ffc5c`。
- 类别约束矩形匹配最大差 5E-7；超过六位小数舍入的匹配框 0；以下列出所有未匹配项（不能推定哪一侧标签正确）：
  - V独有行1：class=1 coords=(0.945312 0.155078 0.075000 0.075000)
  - V独有行2：class=1 coords=(0.371289 0.478906 0.060547 0.062500)
- 除上述结构差异外，成功同类匹配的共同框各坐标差均<=5e-7。

## 3. 文件名来源分组统计
- 正则：`^(MVI_\d+)_img(\d+)_jpg\.rf\.([0-9a-f]+)\.[^.]+$`；8834/8834图片名解析成功；MVI完整编号为视频命名分组，MVI+原img编号为原始帧命名标识，数字差仅是命名帧号差。
- 视频编号共100；train覆盖100，val覆盖99；**99个编号跨train/val**，涉及train7827/7852张、val982/982张。只有 MVI_20032 为train独占25张；val独占视频0。完整计数见下表。
- 同原始帧命名标识跨组共16组、16个T×V配对，均为上述完全相同SHA的16对。**同原始帧标识跨组而图片SHA不同：0组、0对、0张。**
- 完全重复集中于 MVI_40751 的5对，以及 MVI_40752 的11对。
| MVI编号 | train图数 | val图数 |
|---|---:|---:|
| MVI_20011 | 37 | 8 |
| MVI_20012 | 50 | 9 |
| MVI_20032 | 25 | 0 |
| MVI_20033 | 56 | 2 |
| MVI_20034 | 36 | 3 |
| MVI_20035 | 46 | 6 |
| MVI_20051 | 52 | 9 |
| MVI_20052 | 38 | 8 |
| MVI_20061 | 41 | 6 |
| MVI_20062 | 48 | 8 |
| MVI_20063 | 39 | 3 |
| MVI_20064 | 32 | 3 |
| MVI_20065 | 60 | 9 |
| MVI_39031 | 72 | 12 |
| MVI_39051 | 58 | 4 |
| MVI_39211 | 93 | 10 |
| MVI_39271 | 80 | 7 |
| MVI_39311 | 83 | 15 |
| MVI_39361 | 129 | 16 |
| MVI_39371 | 80 | 7 |
| MVI_39401 | 80 | 17 |
| MVI_39501 | 27 | 4 |
| MVI_39511 | 21 | 6 |
| MVI_39761 | 76 | 8 |
| MVI_39771 | 26 | 5 |
| MVI_39781 | 107 | 13 |
| MVI_39801 | 65 | 3 |
| MVI_39811 | 37 | 1 |
| MVI_39821 | 57 | 5 |
| MVI_39851 | 82 | 11 |
| MVI_39861 | 44 | 3 |
| MVI_39931 | 59 | 4 |
| MVI_40131 | 84 | 9 |
| MVI_40141 | 90 | 10 |
| MVI_40152 | 100 | 3 |
| MVI_40161 | 95 | 10 |
| MVI_40162 | 103 | 14 |
| MVI_40171 | 60 | 9 |
| MVI_40172 | 122 | 25 |
| MVI_40181 | 107 | 19 |
| MVI_40191 | 145 | 20 |
| MVI_40192 | 134 | 15 |
| MVI_40201 | 44 | 8 |
| MVI_40204 | 73 | 13 |
| MVI_40211 | 115 | 9 |
| MVI_40212 | 91 | 8 |
| MVI_40213 | 90 | 12 |
| MVI_40241 | 131 | 17 |
| MVI_40243 | 82 | 11 |
| MVI_40244 | 80 | 15 |
| MVI_40701 | 63 | 7 |
| MVI_40711 | 63 | 6 |
| MVI_40712 | 136 | 19 |
| MVI_40714 | 58 | 10 |
| MVI_40732 | 131 | 13 |
| MVI_40742 | 105 | 7 |
| MVI_40743 | 86 | 10 |
| MVI_40751 | 98 | 12 |
| MVI_40752 | 148 | 19 |
| MVI_40761 | 114 | 21 |
| MVI_40762 | 108 | 8 |
| MVI_40763 | 79 | 18 |
| MVI_40771 | 91 | 14 |
| MVI_40772 | 71 | 4 |
| MVI_40773 | 54 | 4 |
| MVI_40774 | 54 | 6 |
| MVI_40775 | 56 | 6 |
| MVI_40792 | 112 | 18 |
| MVI_40793 | 99 | 15 |
| MVI_40851 | 72 | 18 |
| MVI_40852 | 59 | 4 |
| MVI_40853 | 69 | 10 |
| MVI_40854 | 57 | 3 |
| MVI_40855 | 69 | 9 |
| MVI_40863 | 111 | 12 |
| MVI_40864 | 95 | 11 |
| MVI_40871 | 90 | 12 |
| MVI_40891 | 88 | 11 |
| MVI_40892 | 111 | 16 |
| MVI_40901 | 67 | 7 |
| MVI_40902 | 53 | 10 |
| MVI_40903 | 64 | 7 |
| MVI_40904 | 83 | 9 |
| MVI_40905 | 94 | 8 |
| MVI_40962 | 101 | 14 |
| MVI_40963 | 83 | 6 |
| MVI_40981 | 101 | 16 |
| MVI_40991 | 79 | 13 |
| MVI_40992 | 103 | 14 |
| MVI_41063 | 92 | 10 |
| MVI_41073 | 105 | 14 |
| MVI_63521 | 119 | 18 |
| MVI_63525 | 47 | 3 |
| MVI_63544 | 64 | 5 |
| MVI_63552 | 69 | 10 |
| MVI_63553 | 77 | 11 |
| MVI_63554 | 85 | 11 |
| MVI_63561 | 64 | 5 |
| MVI_63562 | 80 | 8 |
| MVI_63563 | 93 | 8 |
## 4. 两张空标签图定位
### MVI_40172_img00696
- 图片：`val/images/MVI_40172_img00696_jpg.rf.71ed1f8014b8b30d31b3650a5593c833.jpg`；MVI=MVI_40172，img=00696；640×640 RGB/JPEG，60300 bytes；SHA256=`856b48d42bdbd3bf767fb8f057dd84557a7054c9b760601b6b56170e0df9e5da`。
- 标签：`val/labels/MVI_40172_img00696_jpg.rf.71ed1f8014b8b30d31b3650a5593c833.txt`；0 bytes、0框，SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- train同图片SHA：0；train同原始帧命名标识：0。
- both范围同视频最近非空标签图，帧号绝对差=5：
  - `train/images/MVI_40172_img00691_jpg.rf.d9495527c45e4f7be660d107b7a9f764.jpg`（6框，class计数{'1': 6}）
  - `train/images/MVI_40172_img00701_jpg.rf.23050b913a3ff3a1e6c72f0062a31ff5.jpg`（5框，class计数{'1': 5}）
  - before最近：差=5；`train/images/MVI_40172_img00691_jpg.rf.d9495527c45e4f7be660d107b7a9f764.jpg` / 6框
  - after最近：差=5；`train/images/MVI_40172_img00701_jpg.rf.23050b913a3ff3a1e6c72f0062a31ff5.jpg` / 5框
- train范围同视频最近非空标签图，帧号绝对差=5：
  - `train/images/MVI_40172_img00691_jpg.rf.d9495527c45e4f7be660d107b7a9f764.jpg`（6框，class计数{'1': 6}）
  - `train/images/MVI_40172_img00701_jpg.rf.23050b913a3ff3a1e6c72f0062a31ff5.jpg`（5框，class计数{'1': 5}）
  - before最近：差=5；`train/images/MVI_40172_img00691_jpg.rf.d9495527c45e4f7be660d107b7a9f764.jpg` / 6框
  - after最近：差=5；`train/images/MVI_40172_img00701_jpg.rf.23050b913a3ff3a1e6c72f0062a31ff5.jpg` / 5框
- val范围同视频最近非空标签图，帧号绝对差=31：
  - `val/images/MVI_40172_img00665_jpg.rf.c6c7115e88f2e0317b64f0cf10a34d7f.jpg`（2框，class计数{'1': 2}）
  - before最近：差=31；`val/images/MVI_40172_img00665_jpg.rf.c6c7115e88f2e0317b64f0cf10a34d7f.jpg` / 2框
  - after最近：差=147；`val/images/MVI_40172_img00843_jpg.rf.7f90d48055ebacafe6187f5523008323.jpg` / 2框
- 上述距离仅命名帧编号差；未用图像内容判断是否真实相邻，不作漏标或正确负样本结论。
### MVI_40172_img02570
- 图片：`val/images/MVI_40172_img02570_jpg.rf.3a486438eecb4812275bd2958a8b10c4.jpg`；MVI=MVI_40172，img=02570；640×640 RGB/JPEG，59236 bytes；SHA256=`6288332d0a15e4bbec5be7734ec5b99371d5ade8945193fac55b12ae644bb310`。
- 标签：`val/labels/MVI_40172_img02570_jpg.rf.3a486438eecb4812275bd2958a8b10c4.txt`；0 bytes、0框，SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- train同图片SHA：0；train同原始帧命名标识：0。
- both范围同视频最近非空标签图，帧号绝对差=1：
  - `train/images/MVI_40172_img02571_jpg.rf.c035b0cd11eedac6d03ad4b387461604.jpg`（5框，class计数{'1': 5}）
  - before最近：差=4；`train/images/MVI_40172_img02566_jpg.rf.cb8fea944d03100fc229ba9aa7b9b09c.jpg` / 5框
  - after最近：差=1；`train/images/MVI_40172_img02571_jpg.rf.c035b0cd11eedac6d03ad4b387461604.jpg` / 5框
- train范围同视频最近非空标签图，帧号绝对差=1：
  - `train/images/MVI_40172_img02571_jpg.rf.c035b0cd11eedac6d03ad4b387461604.jpg`（5框，class计数{'1': 5}）
  - before最近：差=4；`train/images/MVI_40172_img02566_jpg.rf.cb8fea944d03100fc229ba9aa7b9b09c.jpg` / 5框
  - after最近：差=1；`train/images/MVI_40172_img02571_jpg.rf.c035b0cd11eedac6d03ad4b387461604.jpg` / 5框
- val范围同视频最近非空标签图，帧号绝对差=24：
  - `val/images/MVI_40172_img02546_jpg.rf.d801edd0f7cbedc8eb62d51955cbb72c.jpg`（1框，class计数{'1': 1}）
  - before最近：差=24；`val/images/MVI_40172_img02546_jpg.rf.d801edd0f7cbedc8eb62d51955cbb72c.jpg` / 1框
  - after最近：差=None；不存在
- 上述距离仅命名帧编号差；未用图像内容判断是否真实相邻，不作漏标或正确负样本结论。
## 5. 风险证据与未验证范围
- **已证实**验证982张中16张（16/982=1.6293%）与训练图片字节和像素完全相同，构成跨划分输入重叠/验证独立性受限的直接证据；不能因文件名不同而忽略。
- **已证实**16对标签均不精确相同：4对仅坐标舍入相容；12对出现不同框数量/类别结构，其中698、1286还包含超过舍入的几何/类别差异。不能把它们当作“图像和监督完全相同的16对”，也不能据此指定哪边漏标、错标。
- **命名来源证据**：99个视频编号跨组，提示可能存在同源视频相关性；不能升级为已核实上游下载原件、真实视频身份、抽帧顺序/时间间隔或真正相邻帧视觉内容。
- 未读/核验上游原始下载数据或视频；未读取测试集；未检测全量重编码同图、感知近重复或视觉相邻帧；未确认标签语义正确性、漏标/负样本真伪；未估算上述问题对任何指标的方向或幅度；未建议/执行具体删样本或重划分决策。
- 原始标注是坐标数据，不是物体身份。几何最小代价匹配仅机械比较工具；跨类关联尤其不能称为已证实的同一实例重标注。
## 6. 归档交付文件
- 本文：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\reports\data-audit\CLEANING_AUDIT.md`
- 完整JSON（逐匹配行/全部差值/未匹配项/空图前后最近项）：`E:\kaggle\3LC_Multi_Vehicle_detection_Challenge\.worktrees\score-search-20261019-150516\reports\data-audit\cleaning_audit.json`
- 三个可复现脚本：cleaning_audit.py、cleaning_audit_labels.py、cleaning_audit_report.py；均在本报告所在目录。主agent目视结论与待处理清单另见SUMMARY.md。
