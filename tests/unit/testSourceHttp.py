"""Contratos de la capa HTTP con handlers simulados, sin persistencia ni LLM."""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from starlette.routing import Match

from aquai.application.dtos.requests.listSources import ListSourcesRequest
from aquai.application.dtos.requests.reviewSource import SourceDecisionRequest
from aquai.application.dtos.requests.uploadSource import UploadSourceRequest
from aquai.application.errors import (
    FileTooLargeError,
    SourceConflictError,
    SourceDecisionError,
    SourceNotFoundError,
    SourcesUnavailableError,
    TabularInputError,
)
from aquai.application.useCases.sources import (
    GetSourceHandler,
    ListSourcesHandler,
    ReviewSourceHandler,
    UploadSourceHandler,
)
from aquai.infrastructure.providers.sources import SourceRuntime
from aquai.infrastructure.workers.sourceWorker import SourceWorker
from aquai.main import create_app


class SourceHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_id = uuid4()
        self.sources = SourceRuntime(
            upload=Mock(spec=UploadSourceHandler),
            list=Mock(spec=ListSourcesHandler),
            get=Mock(spec=GetSourceHandler),
            review=Mock(spec=ReviewSourceHandler),
            worker=Mock(spec=SourceWorker),
        )
        self.sources.upload.max_upload_bytes = 64
        self.accepted = {
            "id": self.source_id,
            "processing_status": "uploaded",
            "validation_status": "pending",
        }
        self.item = dict(
            self.accepted,
            name="data.csv",
            file_type="csv",
            file_size_bytes=8,
            created_at=datetime(2026, 10, 12, tzinfo=timezone.utc),
            is_active=False,
        )
        self.sources.upload.handle.return_value = self.accepted
        self.sources.list.handle.return_value = {
            "items": [self.item], "total": 1, "page": 1, "page_size": 20,
        }
        self.sources.get.handle.return_value = dict(
            self.item, stage="uploaded", revision=0, queryable=False,
        )
        self.sources.review.handle.return_value = dict(
            self.accepted, review_id="review-1", revision=1, duplicate=False,
        )
        self.client = TestClient(create_app(sources=self.sources))
        self.addCleanup(self.client.close)

    def upload(self, content: bytes = b"name\nAna", **kwargs):
        return self.client.post("/sources", files={"file": ("data.csv", content)}, **kwargs)

    def decide(self, **kwargs):
        return self.client.post(
            f"/sources/{self.source_id}/reviews/review-1/decisions",
            json=kwargs or {"action": "correct_transformations", "values": {"transformations": []}},
        )

    def test_upload_delegates_original_and_serializes_only_the_response_model(self) -> None:
        self.sources.upload.handle.return_value = dict(self.accepted, internal="private")
        response = self.upload()
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json(), dict(self.accepted, id=str(self.source_id)))
        self.sources.upload.handle.assert_awaited_once_with(UploadSourceRequest("data.csv", b"name\nAna"))

    def test_listing_delegates_pagination_and_filters_and_excludes_file_bytes(self) -> None:
        self.sources.list.handle.return_value["items"][0] = dict(self.item, file_content=b"private")
        self.sources.list.handle.return_value.update(page=2, page_size=10)
        response = self.client.get("/sources", params={
            "page": 2, "page_size": 10, "name": "data",
            "processing_status": "review_required", "validation_status": "pending",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.sources.list.handle.assert_awaited_once_with(ListSourcesRequest(
            page=2, page_size=10, name="data", processing_status="review_required", validation_status="pending",
        ))
        result = response.json()
        self.assertEqual((result["page"], result["page_size"]), (2, 10))
        self.assertEqual(set(result["items"][0]), {
            "id", "name", "file_type", "file_size_bytes", "created_at", "isActive",
            "processing_status", "validation_status",
        })
        self.assertFalse(result["items"][0]["isActive"])

    def test_detail_delegates_uuid_and_preserves_findings_and_response_defaults(self) -> None:
        finding = {"id": "finding-1", "kind": "technical", "message": "No se pudo leer el archivo."}
        self.sources.get.handle.return_value.update(findings=[finding], file_content=b"private")
        response = self.client.get(f"/sources/{self.source_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.sources.get.handle.assert_awaited_once_with(self.source_id)
        result = response.json()
        self.assertEqual(result["findings"], [finding])
        self.assertEqual(result["reviews"], [])
        self.assertEqual(result["preview"], [])
        self.assertFalse(result["queryable"])
        self.assertNotIn("file_content", result)

    def test_decision_delegates_structured_values_and_returns_the_existing_contract(self) -> None:
        response = self.decide()
        self.assertEqual(response.status_code, 202, response.text)
        self.sources.review.handle.assert_awaited_once_with(
            self.source_id, "review-1", SourceDecisionRequest(
                action="correct_transformations", values={"transformations": []},
            ),
        )
        self.assertEqual(response.json(), dict(
            self.accepted, id=str(self.source_id), review_id="review-1", revision=1, duplicate=False,
        ))

    def test_file_only_upload_rejects_extra_fields_query_controls_and_multiple_files(self) -> None:
        for control in ("model", "rule_profile", "known_rules"):
            for location in ("data", "params"):
                with self.subTest(control=control, location=location):
                    response = self.upload(**{location: {control: "extra"}})
                    self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/sources", files=[
            ("file", ("one.csv", b"a\n1")), ("file", ("two.csv", b"a\n2")),
        ])
        self.assertEqual(response.status_code, 400, response.text)
        response = self.client.post("/sources", files=[
            ("file", ("one.csv", b"a\n1")), ("other", ("two.csv", b"a\n2")),
        ])
        self.assertEqual(response.status_code, 400, response.text)
        self.sources.upload.handle.assert_not_awaited()

    def test_missing_file_and_invalid_pagination_status_uuid_or_decision_return_422(self) -> None:
        requests = [
            self.client.post("/sources"),
            self.client.post("/sources", files={"other": ("data.csv", b"a\n1")}),
            self.client.get("/sources", params={"page": 0}),
            self.client.get("/sources", params={"page_size": 101}),
            self.client.get("/sources", params={"name": "x" * 256}),
            self.client.get("/sources", params={"processing_status": "unknown"}),
            self.client.get("/sources", params={"validation_status": "unknown"}),
            self.client.get("/sources/not-a-uuid"),
            self.decide(action="correct_transformations", values={}, extra=True),
            self.decide(action="", values={}),
            self.decide(action="correct_transformations", values=[]),
        ]
        for response in requests:
            with self.subTest(url=str(response.request.url), body=response.request.content):
                self.assertEqual(response.status_code, 422, response.text)
        for operation in (
            self.sources.upload.handle,
            self.sources.list.handle,
            self.sources.get.handle,
            self.sources.review.handle,
        ):
            operation.assert_not_awaited()

    def test_upload_rejects_oversize_and_accepts_the_exact_limit(self) -> None:
        self.sources.upload.max_upload_bytes = 5
        response = self.upload(b"123456")
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(response.json(), {"detail": "El archivo supera el tamaño máximo permitido."})
        self.sources.upload.handle.assert_not_awaited()
        self.assertEqual(self.upload(b"12345").status_code, 202)
        self.sources.upload.handle.assert_awaited_once_with(UploadSourceRequest("data.csv", b"12345"))

    def test_multipart_files_close_after_success_rejection_and_handler_failure(self) -> None:
        closed = []
        original_close = UploadFile.close

        async def track_close(file):
            closed.append(file)
            await original_close(file)

        with patch.object(UploadFile, "close", new=track_close):
            self.assertEqual(self.upload().status_code, 202)
            self.assertEqual(self.upload(data={"model": "extra"}).status_code, 400)
            self.sources.upload.handle.side_effect = TabularInputError("Archivo inválido.")
            self.assertEqual(self.upload().status_code, 400)
            self.assertEqual(self.client.post("/sources", files=[
                ("file", ("one.csv", b"a\n1")), ("file", ("two.csv", b"a\n2")),
            ]).status_code, 400)
        self.assertGreaterEqual(len({id(file) for file in closed}), 5)
        self.assertTrue(all(file.file.closed for file in closed))

    def test_application_errors_keep_status_and_detail_on_each_endpoint(self) -> None:
        operations = [
            (self.sources.upload.handle, self.upload),
            (self.sources.list.handle, lambda: self.client.get("/sources")),
            (self.sources.get.handle, lambda: self.client.get(f"/sources/{self.source_id}")),
            (self.sources.review.handle, self.decide),
        ]
        errors = [
            (TabularInputError, 400), (SourceDecisionError, 400),
            (SourceNotFoundError, 404), (SourceConflictError, 409),
            (FileTooLargeError, 413), (SourcesUnavailableError, 503),
        ]
        for operation, request in operations:
            for error_type, expected_status in errors:
                with self.subTest(operation=operation._mock_name, error=error_type.__name__):
                    operation.side_effect = error_type("Explicación del error.")
                    response = request()
                    self.assertEqual(response.status_code, expected_status, response.text)
                    self.assertEqual(response.json(), {"detail": "Explicación del error."})
            operation.side_effect = None

    def test_unavailable_runtime_returns_503_without_executing_an_operation(self) -> None:
        del self.client.app.state.sources
        for request in (
            self.upload,
            lambda: self.client.get("/sources"),
            lambda: self.client.get(f"/sources/{self.source_id}"),
            self.decide,
        ):
            response = request()
            self.assertEqual(response.status_code, 503, response.text)
            self.assertEqual(response.json(), {"detail": "La gestión de fuentes no está disponible."})

    def test_unexpected_technical_errors_are_not_converted_into_user_questions(self) -> None:
        self.sources.get.handle.side_effect = RuntimeError("Fallo técnico inesperado.")
        with TestClient(self.client.app, raise_server_exceptions=False) as client:
            response = client.get(f"/sources/{self.source_id}")
        self.assertEqual(response.status_code, 500)

    def test_lifespan_starts_and_stops_only_the_worker(self) -> None:
        with TestClient(self.client.app) as client:
            self.sources.worker.start.assert_awaited_once_with()
            self.sources.worker.stop.assert_not_awaited()
            self.assertEqual(client.get("/openapi.json").status_code, 200)
        self.sources.worker.stop.assert_awaited_once_with()
        for handler in (self.sources.upload, self.sources.list, self.sources.get, self.sources.review):
            handler.handle.assert_not_awaited()

    def test_openapi_uses_response_models_and_has_only_the_four_source_operations(self) -> None:
        schema = self.client.get("/openapi.json").json()
        paths = schema["paths"]
        self.assertEqual(set(paths), {
            "/sources", "/sources/{id}", "/sources/{id}/reviews/{review_id}/decisions",
        })
        operations = [
            (paths["/sources"]["post"], "202", "SourceAcceptedResponse"),
            (paths["/sources"]["get"], "200", "SourceListResponse"),
            (paths["/sources/{id}"]["get"], "200", "SourceDetailResponse"),
            (paths["/sources/{id}/reviews/{review_id}/decisions"]["post"], "202", "SourceDecisionResponse"),
        ]
        for operation, status, model in operations:
            with self.subTest(model=model):
                response = operation["responses"][status]["content"]["application/json"]
                self.assertEqual(response["schema"], {"$ref": f"#/components/schemas/{model}"})
                for documented in operation["responses"].values():
                    self.assertTrue(documented["description"])
                    for media in documented.get("content", {}).values():
                        self.assertNotIn("example", media)
                        self.assertNotIn("examples", media)
        for operation in (paths["/sources"]["get"], paths["/sources/{id}"]["get"]):
            self.assertNotIn("requestBody", operation)
        detail_model = schema["components"]["schemas"]["SourceDetailResponse"]
        for field, model in (("blocks", "SourceBlock"), ("preview", "SourcePreview"),
                             ("findings", "SourceFinding"), ("reviews", "SourceReview"),
                             ("decisions", "SourceDecision"), ("technical_errors", "SourceTechnicalError")):
            self.assertEqual(detail_model["properties"][field]["items"], {"$ref": f"#/components/schemas/{model}"})
        review_model = schema["components"]["schemas"]["SourceReview"]
        self.assertEqual(review_model["properties"]["options"]["items"], {"$ref": "#/components/schemas/SourceReviewOption"})
        upload = paths["/sources"]["post"]
        self.assertNotIn("parameters", upload)
        body = upload["requestBody"]["content"]["multipart/form-data"]["schema"]
        body_model = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(set(body_model["properties"]), {"file"})
        self.assertEqual(body_model["required"], ["file"])
        self.assertNotIn("TabularEvaluationResponse", schema["components"]["schemas"])
        scope = {"type": "http", "path": "/experimental/tabular/evaluate", "root_path": "", "method": "POST"}
        self.assertTrue(all(route.matches(scope)[0] == Match.NONE for route in self.client.app.routes))
        self.assertEqual(self.client.post("/experimental/tabular/evaluate").status_code, 404)

    def test_openapi_contracts_have_no_manual_documentation_examples(self) -> None:
        schema = self.client.get("/openapi.json").json()
        pending = [schema]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                self.assertNotIn("example", value)
                self.assertNotIn("examples", value)
                self.assertNotIn("openapi_examples", value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)


if __name__ == "__main__":
    unittest.main()
