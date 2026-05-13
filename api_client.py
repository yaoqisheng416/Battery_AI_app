# -*- coding: utf-8 -*-
import requests

API_BASE = "http://127.0.0.1:8000"


def create_task(endpoint, payload):

    try:

        response = requests.post(
            f"{API_BASE}{endpoint}",
            json=payload,
            timeout=60,
        )

        return response.json()

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


def query_task(task_id):

    response = requests.get(
        f"{API_BASE}/task/query/{task_id}"
    )

    return response.json()


def get_model_versions():
    try:

        response = requests.get(
            f"{API_BASE}/models/versions"
        )

        data = response.json()

        return data.get("versions", [])

    except:

        return []


def query_all_tasks():
    url = f"{API_BASE}/tasks"

    response = requests.get(
        url,
        timeout=30,
    )

    return response.json()