# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import threading
import zipfile
import logging
from datetime import datetime
from http.client import HTTPException
import uvicorn

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse

from backend.schemas import Stage4Request, cbdGenerateRequest, fitParameterRequest, largeVolumeGenerateRequest, \
    localConditionsGenerateRequest

from backend.core.task_manager import (
    create_task,
    get_task, TASK_STORE,
)

from backend.tasks.stage4_task import run_stage4_task
from backend.tasks.stage5_task import run_large_volume_generate_task, run_local_conditions_generate_task
from backend.tasks.stage6_task import run_stage6_cbd_fit_task, run_stage6_cbd_generate_task

logger = logging.getLogger("cbd_w_fitting_service")

if not logger.handlers:
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)

app = FastAPI()

# =========================================================
# cors
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return JSONResponse(
        content={
            "message": "Welcome to the BA-API!",
            "docs": "Visit /docs for API documentation",
            "available_endpoints": ["/tasks", "/task/{task_id}"]
        },
        status_code=200
    )


@app.get("/tasks")
def get_all_tasks():
    return list(
        TASK_STORE.values()
    )


# =========================================================
# Create Stage4 generate_structure_from_condition Task API
# =========================================================
@app.post("/stage4/generate_structure_from_condition")
def create_stage4_task(
        request: Stage4Request
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:
        task_id = create_task(title="Stage4 条件可控的两相结构生成")

    #  打印日志（调试用）
    print(f"[Stage4] 任务ID: {task_id}")
    print(f"[Stage4] VAE路径: {request.vae_path}")
    print(f"[Stage4] LDM路径: {request.ldm_path}")
    print(f"[Stage4] Porosity: {request.porosity}")
    print(f"[Stage4] Tau Z: {request.tau_z}")
    print(f"[Stage4] Surface Area: {request.surface_area}")

    thread = threading.Thread(
        target=run_stage4_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Query Task API
# ============================================================
@app.get("/task/query/{task_id}")
def query_task(task_id: str):
    task = get_task(task_id)

    if task is None:
        return {
            "status": "not_found"
        }

    return task


# ============================================================
# Create Stage5 build_large_volume_conditions_from_real Task API
# ============================================================
@app.post("/stage5/local-conditions-generate")
def create_stage5_task(
        request: localConditionsGenerateRequest
):

    # ============================================
    # task_id
    # ============================================
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(
            title="Stage5 从真实体积构建local conditions"
        )

    # ============================================
    # start thread
    # ============================================
    thread = threading.Thread(
        target=run_local_conditions_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    # ============================================
    # response
    # ============================================
    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage5 large-volume-generate Task API
# ============================================================
@app.post("/stage5/large-volume-generate")
def create_stage5_task(
        request: largeVolumeGenerateRequest
):
    # ============================================
    # task_id 处理
    # ============================================
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(
            title="Stage5 224³大体积生成"
        )

    # ============================================
    # start thread
    # ============================================
    thread = threading.Thread(
        target=run_large_volume_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    # ============================================
    # response
    # ============================================
    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage6 cbd-generate Task API
# ============================================================
@app.post("/stage6/cbd-generate")
def create_stage6_task(
        request: cbdGenerateRequest
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(

            title="Stage6 CBD三相电极结构生成"
        )

    thread = threading.Thread(
        target=run_stage6_cbd_generate_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# ============================================================
# Create Stage6 fit_cbd_spreading_parameter Task API
# ============================================================
@app.post("/stage6/fit-cbd-spreading-parameter")
def create_stage6_task(
        request: fitParameterRequest
):
    if request.task_id is not None \
            and str(request.task_id).strip() != "":

        task_id = request.task_id

    else:

        task_id = create_task(

            title="Stage6 CBD参数拟合"
        )

    thread = threading.Thread(
        target=run_stage6_cbd_fit_task,
        args=(task_id, request),
        daemon=True,
    )

    thread.start()

    return {
        "task_id": task_id,
        "status": "queued",
    }


# 模型选择
def get_base_dir():
    """智能判断：打包后用exe目录，本地用脚本目录"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)  # 打包后exe目录
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 本地模式: 从脚本位置向上找项目根目录


@app.get("/health")
def health(SERVER_READY=True):
    return {
        "ready": SERVER_READY
    }


@app.get("/models/versions")
def get_model_versions():
    BASE_DIR = get_base_dir()

    # 智能切换路径
    if getattr(sys, 'frozen', False):
        vae_dir = os.path.join(BASE_DIR, "checkpoints")
        ldm_dir = os.path.join(BASE_DIR, "ldm_checkpoints")
        print(f"[打包模式] VAE: {vae_dir}")
        print(f"[打包模式] LDM: {ldm_dir}")
    else:
        vae_dir = os.path.join(BASE_DIR, "checkpoints")
        ldm_dir = os.path.join(BASE_DIR, "ldm_checkpoints")
        print(f"[本地模式] VAE: {vae_dir}")
        print(f"[本地模式] LDM: {ldm_dir}")

    # 扫描 VAE
    vae_models = []
    if os.path.exists(vae_dir):
        for file_name in sorted(os.listdir(vae_dir), key=lambda x: os.path.getmtime(os.path.join(vae_dir, x)),
                                reverse=True):
            if file_name.endswith(".ckpt"):
                vae_models.append({
                    "file_name": file_name,
                    "full_path": os.path.join(vae_dir, file_name),
                    "create_time": datetime.fromtimestamp(os.path.getmtime(os.path.join(vae_dir, file_name))).strftime(
                        "%Y-%m-%d %H:%M:%S"),
                })
        print(f"[OK] 找到 {len(vae_models)} 个VAE模型")
    else:
        print(f"[警告] VAE目录不存在：{vae_dir}")

    # 扫描 LDM
    ldm_models = []
    if os.path.exists(ldm_dir):
        for file_name in sorted(os.listdir(ldm_dir), key=lambda x: os.path.getmtime(os.path.join(ldm_dir, x)),
                                reverse=True):
            if file_name.endswith(".ckpt"):
                ldm_models.append({
                    "file_name": file_name,
                    "full_path": os.path.join(ldm_dir, file_name),
                    "create_time": datetime.fromtimestamp(os.path.getmtime(os.path.join(ldm_dir, file_name))).strftime(
                        "%Y-%m-%d %H:%M:%S"),
                })
        print(f"[OK] 找到 {len(ldm_models)} 个LDM模型")
    else:
        print(f"[警告] LDM目录不存在：{ldm_dir}")

    return {
        "vae_models": vae_models,
        "ldm_models": ldm_models,
        "base_dir": BASE_DIR,
        "mode": "frozen" if getattr(sys, 'frozen', False) else "dev",
    }


# ========================================================
#  上传最优二相结构
# ========================================================
@app.post("/upload/b2ps")
async def upload_best_two_phase_structure(

        file: UploadFile = File(...)
):
    # ====================================================
    # task id
    # ====================================================
    task_id = create_task(
        title="Stage6 CBD三相电极结构生成"
    )

    # ====================================================
    # workspace
    # ====================================================
    task_root = os.path.join(

        "workspace",
        "tasks",
        task_id,
    )

    input_dir = os.path.join(
        task_root,
        "input"
    )

    os.makedirs(
        input_dir,
        exist_ok=True
    )

    # ====================================================
    # save file
    # ====================================================
    save_path = os.path.join(

        input_dir,

        file.filename
    )

    with open(save_path, "wb") as f:
        content = await file.read()

        f.write(content)

    return {

        "success": True,

        "task_id": task_id,

        "input_file": save_path,
    }


# ========================================================
#  下载stage6 cbd生成结果
# ========================================================
@app.get("/download/{task_id}/{file_name}")
def download_file(

        task_id: str,

        file_name: str,
):
    file_path = os.path.join(

        "workspace",
        "tasks",
        task_id,
        "out_put",
        file_name,
    )

    if not os.path.exists(file_path):
        return {
            "error": "file not found"
        }

    return FileResponse(

        file_path,

        filename=file_name,
    )


# 获取处理结果展示
@app.get("/task/results/list")
def list_task_results(task_id: str = Query(...)):
    base_dir = os.path.join(
        "workspace",
        "tasks",
        task_id,
        "out_put"
    )

    print("base_dir:", base_dir)

    if not os.path.exists(base_dir):
        return {
            "files": [],
            "debug": "dir not exists",
            "path": base_dir
        }

    results = []

    for root, _, files in os.walk(base_dir):

        for f in files:
            full = os.path.join(root, f)

            results.append({
                "name": f,
                "path": full,
                "type": "file"
            })

    return {
        "files": results
    }


# 下载单个文件
@app.get("/download/file")
def download_file(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "file not found")

    if os.path.isdir(path):
        raise HTTPException(400, "path is directory, use zip download")

    return FileResponse(
        path,
        filename=os.path.basename(path)
    )


# 下载为zip包
@app.get("/download/dir")
def download_dir(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "not found")

    if not os.path.isdir(path):
        raise HTTPException(400, "not a directory")

    tmp_zip = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    )

    with zipfile.ZipFile(tmp_zip.name, "w") as z:

        for root, _, files in os.walk(path):

            for f in files:
                full_path = os.path.join(root, f)

                arcname = os.path.relpath(
                    full_path,
                    path
                )

                z.write(full_path, arcname)

    return FileResponse(
        tmp_zip.name,
        filename=os.path.basename(path) + ".zip"
    )


# ============================================
#  修改入口：打包时用这个函数启动
# ============================================
def start_server():
    """启动后端 API（供 subprocess 调用）"""
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info",
        reload=False,
        log_config=None,
        access_log=False
    )


#  原来的入口（保留，方便单独测试）
if __name__ == "__main__":
    start_server()
