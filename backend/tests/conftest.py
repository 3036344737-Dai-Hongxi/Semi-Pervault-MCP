"""测试进程级防护：先于任何 main 导入锚定 DB 路径到临时目录。

否则 import main 会触发 ~/.pervault 的真实迁移逻辑，测试不得污染用户主目录。
"""

import os
import tempfile

os.environ.setdefault(
    "PERVAULT_DB_PATH",
    os.path.join(tempfile.mkdtemp(prefix="pervault-test-"), "data.db"),
)
