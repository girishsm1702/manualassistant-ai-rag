import json
from pathlib import Path

CASES = json.loads((Path(__file__).parent / "evaluation_cases.json").read_text(encoding="utf-8"))


def test_evaluation_dataset(loaded_service):
    failures = []
    for case in CASES:
        result = loaded_service.ask(
            machine_id=case["machine_id"],
            question=case["question"],
            min_score=case.get("min_score"),
        )
        evidence_text = " ".join(item.chunk.text for item in result.evidence)
        combined = result.answer + " " + evidence_text

        if case["expect_status"] == "insufficient_or_no_foreign_machine":
            if any(item.chunk.machine_id != case["machine_id"] for item in result.evidence):
                failures.append(f"{case['id']}: leaked another machine")
            if case.get("forbidden") and case["forbidden"] in combined:
                failures.append(f"{case['id']}: contained forbidden text {case['forbidden']}")
            continue

        if result.status != case["expect_status"]:
            failures.append(f"{case['id']}: status {result.status} != {case['expect_status']}")
            continue
        if case.get("expect_page") is not None and result.citations:
            pages = [citation.page_number for citation in result.citations]
            if case["expect_page"] not in pages:
                failures.append(f"{case['id']}: expected page {case['expect_page']} in {pages}")
        if case.get("must_contain") and case["must_contain"] not in combined:
            failures.append(f"{case['id']}: missing {case['must_contain']}")
    assert not failures, failures
