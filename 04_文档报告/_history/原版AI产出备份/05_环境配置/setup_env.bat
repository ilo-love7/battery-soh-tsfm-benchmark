@echo off
:: 智诚时序基础模型零样本电池基准 - 环境设置脚本
:: Windows 批处理版本

echo ========================================
echo  智诚时序基础模型项目 - 环境初始化
echo ========================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 未安装或不在 PATH 中！
    echo 请访问 https://www.python.org/downloads/ 下载安装
    pause
    exit /b 1
)

echo [OK] Python 已安装
python --version

echo.
echo [1/3] 创建虚拟环境...
if not exist "venv" (
    python -m venv venv
    echo [OK] 虚拟环境创建成功
) else (
    echo [INFO] 虚拟环境已存在
)

echo.
echo [2/3] 激活虚拟环境...
call venv\Scripts\activate.bat

echo.
echo [3/3] 安装核心依赖包...
pip install --upgrade pip
pip install numpy pandas scipy torch matplotlib seaborn scikit-learn

echo.
echo [可选] 是否安装完整依赖？(y/n)
set /p install_full=
if /i "%install_full%"=="y" (
    echo 正在安装完整依赖...
    pip install -r requirements.txt
)

echo.
echo ========================================
echo  环境设置完成！
echo  请运行：venv\Scripts\activate.bat 激活环境
echo ========================================
echo.
pause
