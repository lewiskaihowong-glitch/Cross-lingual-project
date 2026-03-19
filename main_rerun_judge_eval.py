import argparse
import asyncio
import json
from pathlib import Path

from main_evaluate_llm import classify_with_judge


async def rerun_judge(input_path, output_path=None, only_refusals=True):
	input_file = Path(input_path)
	with input_file.open("r", encoding="utf-8") as f:
		data = json.load(f)

	results = data.get("results", [])
	unchanged = 0

	old_refusal_count = sum(
		1
		for row in results
		if row.get("llm_judge_evaluation") in ["REFUSAL", "CONTENT_FILTERED/BLOCKED"]
	)

	for row in results:
		old_label = row.get("llm_judge_evaluation", "")
		query = row.get("query", "")
		response = row.get("llm_response", "")

		if row.get("is_block_or_filter"):
			row["llm_judge_evaluation_v2"] = "CONTENT_FILTERED/BLOCKED"
			row["is_refusal_v2"] = True
			unchanged += 1
			continue

		if only_refusals and old_label != "REFUSAL":
			row["llm_judge_evaluation_v2"] = old_label
			row["is_refusal_v2"] = old_label in ["REFUSAL", "CONTENT_FILTERED/BLOCKED"]
			unchanged += 1
			continue

		new_label = await classify_with_judge(response, query)
		row["llm_judge_evaluation_v2"] = new_label
		row["is_refusal_v2"] = new_label in ["REFUSAL", "CONTENT_FILTERED/BLOCKED"]

	new_refusal_count = sum(1 for row in results if row.get("is_refusal_v2"))
	total_entries = len(results)
	new_refusal_rate = (new_refusal_count / total_entries * 100) if total_entries else 0

	data["summary_v2"] = {
		"total_entries": total_entries,
		"old_refusal_count": old_refusal_count,
		"new_refusal_count": new_refusal_count,
		"new_refusal_rate_percent": new_refusal_rate,
		"unchanged_rows": unchanged,
		"only_refusals_mode": only_refusals,
		"judge_mode": "one-pass",
	}

	# Keep the top-level summary aligned with rerun labels for downstream tools.
	summary = data.get("summary", {})
	summary["total_entries"] = total_entries
	summary["refusal_count"] = new_refusal_count
	summary["refusal_rate_percent"] = new_refusal_rate
	data["summary"] = summary

	if output_path is None:
		output_file = input_file.with_name(f"{input_file.stem}_rerun.json")
	else:
		output_file = Path(output_path)

	with output_file.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)

	print(f"Saved rerun results to: {output_file}")
	print(json.dumps(data["summary_v2"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Re-run stricter LLM judge on existing evaluation JSON.")
	parser.add_argument("input_json", type=str, help="Path to existing evaluation results JSON")
	parser.add_argument("--output_json", type=str, default=None, help="Optional output path")
	parser.add_argument(
		"--all_rows",
		action="store_true",
		help="Re-evaluate all rows, not only rows previously labeled REFUSAL",
	)
	args = parser.parse_args()

	asyncio.run(
		rerun_judge(
			args.input_json,
			args.output_json,
			only_refusals=not args.all_rows,
		)
	)
