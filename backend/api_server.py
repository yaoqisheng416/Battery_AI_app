# -*- coding: utf-8 -*-
import os
import tempfile
import threading
import zipfile
from datetime import datetime
from http.client import HTTPException
import uvicorn

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse

from backend.schemas import Stage4Request, cbdGenerateRequest, fitParameterRequest

from backend.core.task_manager import (
    create_task,
    get_task, TASK_STORE,
)

from backend.tasks.stage4_task import run_stage4_task
from backend.tasks.stage6_task import run_stage6_cbd_fit_task, run_stage6_cbd_generate_task

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
    task_id = create_task(title="Stage4 条件可控的两相结构生成")

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
        # print("task_id", task_id)

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
@app.get("/models/versions")
def get_model_versions():
    model_root = "models"

    versions = []

    if not os.path.exists(model_root):
        return {
            "versions": []
        }

    # ========================================================
    # scan versions
    # ========================================================
    for version_name in os.listdir(model_root):

        version_dir = os.path.join(
            model_root,
            version_name,
        )

        if not os.path.isdir(version_dir):
            continue

        # ====================================================
        # ldm
        # ====================================================
        ldm_dir = os.path.join(
            version_dir,
            "ldm"
        )

        ldm_models = []

        if os.path.exists(ldm_dir):

            for file_name in os.listdir(ldm_dir):

                if file_name.endswith(".ckpt"):
                    ldm_models.append({

                        "file_name": file_name,

                        "full_path":
                            os.path.join(
                                ldm_dir,
                                file_name
                            ),
                    })

        # ====================================================
        # vae
        # ====================================================
        vae_dir = os.path.join(
            version_dir,
            "vae"
        )

        vae_models = []

        if os.path.exists(vae_dir):

            for file_name in os.listdir(vae_dir):

                if file_name.endswith(".ckpt"):
                    vae_models.append({

                        "file_name": file_name,

                        "full_path":
                            os.path.join(
                                vae_dir,
                                file_name
                            ),
                    })

        # ====================================================
        # append
        # ====================================================
        versions.append({

            "version": version_name,

            "create_time":
                datetime.fromtimestamp(

                    os.path.getmtime(
                        version_dir
                    )

                ).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "ldm_models": ldm_models,

            "vae_models": vae_models,
        })

    # ========================================================
    # sort
    # ========================================================
    versions = sorted(

        versions,

        key=lambda x: x["create_time"],

        reverse=True,
    )

    return {
        "versions": versions
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
        port=8000,
        log_level="info",
        reload=False
    )


#  原来的入口（保留，方便单独测试）
if __name__ == "__main__":
    start_server()
