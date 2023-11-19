from fastapi import APIRouter

from routers import regression

router = APIRouter(
    prefix='/api')

routers = (regression,)

for r in routers:
    router.include_router(r.router)
