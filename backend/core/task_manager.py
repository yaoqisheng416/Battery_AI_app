import threading
import traceback
import uuid
from datetime import datetime

TASK_STORE = {}


def create_task(title="unknown"):

    task_id = str(uuid.uuid4())

    TASK_STORE[task_id] = {

        "task_id": task_id,

        "title": title,

        "status": "queued",

        "progress": 0,

        "logs": [],

        "result": None,

        "error": None,

        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return task_id


def append_log(task_id, message):

    print(message)

    TASK_STORE[task_id]["logs"].append(str(message))


def update_progress(task_id, progress):

    TASK_STORE[task_id]["progress"] = progress


def set_running(task_id):

    TASK_STORE[task_id]["status"] = "running"


def set_finished(task_id, result):

    TASK_STORE[task_id]["status"] = "finished"

    TASK_STORE[task_id]["progress"] = 100

    TASK_STORE[task_id]["result"] = result


def set_failed(task_id, error):

    TASK_STORE[task_id]["status"] = "failed"

    TASK_STORE[task_id]["error"] = str(error)


def get_task(task_id):

    return TASK_STORE.get(task_id)
