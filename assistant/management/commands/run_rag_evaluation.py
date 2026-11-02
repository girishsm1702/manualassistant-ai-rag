"""Benchmark and evaluate the Grounded RAG Assistant against test cases."""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np
from django.core.management.base import BaseCommand

from assistant.services import ask_question, health_status


class Command(BaseCommand):
    help = "Run the RAG Grounding & Isolation evaluation benchmark suite."

    def add_arguments(self, parser):
        parser.add_argument(
            "--cases-path",
            type=str,
            default=str(Path(__file__).resolve().parent.parent.parent.parent / "tests" / "evaluation_cases.json"),
            help="Path to evaluation_cases.json",
        )

    def handle(self, *args, **options):
        cases_file = Path(options["cases_path"])
        if not cases_file.exists():
            self.stderr.write(f"Cases file not found at {cases_file}")
            return

        cases = json.loads(cases_file.read_text(encoding="utf-8"))
        self.stdout.write(self.style.MIGRATE_HEADING("=== RAG Assistant Evaluation Benchmark ==="))
        
        health = health_status()
        self.stdout.write(
            f"Database: {health['database']['engine']} | Vector Store: {health['vector_backend']} | Chunks: {health['chunk_count']}\n"
        )

        results = []
        latencies = []
        passed_count = 0

        for case in cases:
            case_id = case["id"]
            machine_id = case["machine_id"]
            question = case["question"]
            expected_status = case["expect_status"]

            start_t = time.perf_counter()
            result = ask_question(
                machine_id=machine_id,
                question=question,
                min_score=case.get("min_score"),
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(result.latency_ms or elapsed_ms)

            evidence_text = " ".join(item.chunk.text for item in result.evidence)
            combined = result.answer + " " + evidence_text

            passed = True
            reasons = []

            if expected_status == "insufficient_or_no_foreign_machine":
                if any(item.chunk.machine_id != machine_id for item in result.evidence):
                    passed = False
                    reasons.append("Machine isolation failure: leaked foreign machine chunks")
                if case.get("forbidden") and case["forbidden"] in combined:
                    passed = False
                    reasons.append(f"Contains forbidden text: {case['forbidden']}")
            else:
                if result.status != expected_status:
                    passed = False
                    reasons.append(f"Status mismatch: got {result.status}, expected {expected_status}")
                if case.get("expect_page") is not None and result.citations:
                    pages = [c.page_number for c in result.citations]
                    if case["expect_page"] not in pages:
                        passed = False
                        reasons.append(f"Page {case['expect_page']} not in citations {pages}")
                if case.get("must_contain") and case["must_contain"] not in combined:
                    passed = False
                    reasons.append(f"Missing required phrase: {case['must_contain']}")

            if passed:
                passed_count += 1
                status_str = self.style.SUCCESS("PASS")
            else:
                status_str = self.style.ERROR(f"FAIL ({'; '.join(reasons)})")

            results.append({
                "id": case_id,
                "machine": machine_id,
                "latency_ms": result.latency_ms or elapsed_ms,
                "status": result.status,
                "passed": passed,
            })
            self.stdout.write(f"[{status_str}] {case_id:<20} latency: {elapsed_ms:>6.1f} ms | citations: {len(result.citations)}")

        self.stdout.write("\n" + "=" * 55)
        self.stdout.write(self.style.MIGRATE_HEADING("Benchmark Metrics Summary:"))
        self.stdout.write(f"Total Test Cases   : {len(cases)}")
        self.stdout.write(f"Cases Passed       : {passed_count}/{len(cases)} ({passed_count/len(cases)*100:.1f}%)")
        self.stdout.write(f"Average Latency    : {np.mean(latencies):.1f} ms")
        self.stdout.write(f"P50 Latency (Median): {np.median(latencies):.1f} ms")
        self.stdout.write(f"P95 Latency        : {np.percentile(latencies, 95):.1f} ms")
        self.stdout.write("=" * 55 + "\n")
