"""
Sample backend: Pet Store REST API.

Şəmadan iki versiyası var:
- v1 (default): klassik sxem (name, species, age, status)
- v2: təkamül etmiş sxem (name, kind, age_months, available); sahə adları və endpoint
       yolları dəyişib; self-healing mexanizmi məhz bu cür dəyişiklikləri aşkarlamalı və
       bərpa etməlidir.

İşə salmaq:
    SCHEMA_VERSION=v1 uvicorn sample_backend.main:app --port 8000
    SCHEMA_VERSION=v2 uvicorn sample_backend.main:app --port 8000
"""

from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "v1").lower()

app = FastAPI(
    title="Pet Store API",
    version=SCHEMA_VERSION,
    description=(
        f"AI-əsaslı adaptiv test prototipinin sınaq hədəfi. Cari sxem versiyası: {SCHEMA_VERSION}."
    ),
)


# ---------- v1 sxemi ----------


class PetV1(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    species: str = Field(min_length=1)
    age: int = Field(ge=0, le=50)


class PetV1Out(PetV1):
    id: str
    status: str = "available"


class OrderV1(BaseModel):
    pet_id: str
    quantity: int = Field(ge=1)


class OrderV1Out(OrderV1):
    id: str
    placed_at: float


# ---------- v2 sxemi (təkamül etmiş) ----------


class PetV2(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1)  # species → kind
    age_months: int = Field(ge=0, le=600)  # age (years) → age_months


class PetV2Out(PetV2):
    pet_id: str  # id → pet_id
    available: bool = True  # status: str → available: bool


class OrderV2(BaseModel):
    petId: str  # snake_case → camelCase
    quantity: int = Field(ge=1)


class OrderV2Out(OrderV2):
    orderId: str
    timestamp: int  # float epoch → int epoch ms


# ---------- yaddaş ----------

_pets: dict[str, dict[str, Any]] = {}
_orders: dict[str, dict[str, Any]] = {}


# ---------- v1 endpoint-ləri ----------

if SCHEMA_VERSION == "v1":

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "v1"}

    @app.post("/pets", response_model=PetV1Out, status_code=status.HTTP_201_CREATED)
    def create_pet(pet: PetV1) -> PetV1Out:
        pet_id = str(uuid4())
        record = pet.model_dump() | {"id": pet_id, "status": "available"}
        _pets[pet_id] = record
        return PetV1Out(**record)

    @app.get("/pets", response_model=list[PetV1Out])
    def list_pets() -> list[PetV1Out]:
        return [PetV1Out(**p) for p in _pets.values()]

    @app.get("/pets/{pet_id}", response_model=PetV1Out)
    def get_pet(pet_id: str) -> PetV1Out:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        return PetV1Out(**_pets[pet_id])

    @app.put("/pets/{pet_id}", response_model=PetV1Out)
    def update_pet(pet_id: str, pet: PetV1) -> PetV1Out:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        record = pet.model_dump() | {"id": pet_id, "status": _pets[pet_id]["status"]}
        _pets[pet_id] = record
        return PetV1Out(**record)

    @app.delete("/pets/{pet_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_pet(pet_id: str) -> None:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        del _pets[pet_id]

    @app.post("/orders", response_model=OrderV1Out, status_code=status.HTTP_201_CREATED)
    def create_order(order: OrderV1) -> OrderV1Out:
        if order.pet_id not in _pets:
            raise HTTPException(400, "Unknown pet_id")
        order_id = str(uuid4())
        record = order.model_dump() | {"id": order_id, "placed_at": time.time()}
        _orders[order_id] = record
        return OrderV1Out(**record)

    @app.get("/orders/{order_id}", response_model=OrderV1Out)
    def get_order(order_id: str) -> OrderV1Out:
        if order_id not in _orders:
            raise HTTPException(404, "Order not found")
        return OrderV1Out(**_orders[order_id])


# ---------- v2 endpoint-ləri (təkamül etmiş) ----------

elif SCHEMA_VERSION == "v2":

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "v2"}

    @app.post("/api/v2/pets", response_model=PetV2Out, status_code=status.HTTP_201_CREATED)
    def create_pet(pet: PetV2) -> PetV2Out:
        pet_id = str(uuid4())
        record = pet.model_dump() | {"pet_id": pet_id, "available": True}
        _pets[pet_id] = record
        return PetV2Out(**record)

    @app.get("/api/v2/pets", response_model=list[PetV2Out])
    def list_pets() -> list[PetV2Out]:
        return [PetV2Out(**p) for p in _pets.values()]

    @app.get("/api/v2/pets/{pet_id}", response_model=PetV2Out)
    def get_pet(pet_id: str) -> PetV2Out:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        return PetV2Out(**_pets[pet_id])

    @app.put("/api/v2/pets/{pet_id}", response_model=PetV2Out)
    def update_pet(pet_id: str, pet: PetV2) -> PetV2Out:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        record = pet.model_dump() | {
            "pet_id": pet_id,
            "available": _pets[pet_id]["available"],
        }
        _pets[pet_id] = record
        return PetV2Out(**record)

    @app.delete("/api/v2/pets/{pet_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_pet(pet_id: str) -> None:
        if pet_id not in _pets:
            raise HTTPException(404, "Pet not found")
        del _pets[pet_id]

    @app.post("/api/v2/orders", response_model=OrderV2Out, status_code=status.HTTP_201_CREATED)
    def create_order(order: OrderV2) -> OrderV2Out:
        if order.petId not in _pets:
            raise HTTPException(400, "Unknown petId")
        order_id = str(uuid4())
        record = order.model_dump() | {
            "orderId": order_id,
            "timestamp": int(time.time() * 1000),
        }
        _orders[order_id] = record
        return OrderV2Out(**record)

    @app.get("/api/v2/orders/{order_id}", response_model=OrderV2Out)
    def get_order(order_id: str) -> OrderV2Out:
        if order_id not in _orders:
            raise HTTPException(404, "Order not found")
        return OrderV2Out(**_orders[order_id])

else:
    raise RuntimeError(f"Naməlum SCHEMA_VERSION: {SCHEMA_VERSION!r}")
