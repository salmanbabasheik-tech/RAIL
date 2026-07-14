import os
import json
import tempfile
import logging
import boto3
from flask import Flask, request, jsonify, send_from_directory
from score import main as score_resume

app = Flask(__name__, static_folder="static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JD_MATCH_PROMPT = """You are an expert technical recruiter. Compare this resume against the job description and return ONLY a JSON object (no markdown):

{
  "match_score": 0-100,
  "verdict": "Strong Match" | "Good Match" | "Partial Match" | "Weak Match",
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "matched_experience": ["point1", "point2"],
  "missing_experience": ["point1", "point2"],
  "recommendation": "one paragraph recommendation",
  "hire_recommendation": "Yes" | "Maybe" | "No"
}

JOB DESCRIPTION:
{jd}

RESUME:
{resume}

Return ONLY valid JSON."""


def call_bedrock(prompt: str) -> str:
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    response = client.converse(
        modelId="us.anthropic.claude-opus-4-6-v1",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 0.1},
    )
    return response["output"]["message"]["content"][0]["text"]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/score", methods=["POST"])
def score():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        logger.info(f"Scoring resume: {file.filename}")
        result = score_resume(tmp_path)

        if result is None:
            return jsonify({"error": "Failed to process resume. Please try again."}), 500

        scores = {}
        total = 0
        max_score = 0

        if hasattr(result, "scores") and result.scores:
            for key, val in result.scores.model_dump().items():
                capped = min(val["score"], val["max"])
                scores[key] = {"score": capped, "max": val["max"], "evidence": val["evidence"]}
                total += capped
                max_score += val["max"]

        bonus = 0
        bonus_breakdown = ""
        if hasattr(result, "bonus_points") and result.bonus_points:
            bonus = result.bonus_points.total
            bonus_breakdown = result.bonus_points.breakdown
            total += bonus

        deductions = 0
        deduction_reasons = ""
        if hasattr(result, "deductions") and result.deductions:
            deductions = result.deductions.total
            deduction_reasons = result.deductions.reasons
            total -= deductions

        return jsonify({
            "total_score": round(max(0, min(120, total)), 1),
            "max_score": max_score,
            "scores": scores,
            "bonus_points": {"total": bonus, "breakdown": bonus_breakdown},
            "deductions": {"total": deductions, "reasons": deduction_reasons},
            "key_strengths": result.key_strengths if hasattr(result, "key_strengths") else [],
            "areas_for_improvement": result.areas_for_improvement if hasattr(result, "areas_for_improvement") else [],
        })

    except Exception as e:
        logger.error(f"Error scoring resume: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/match", methods=["POST"])
def match():
    if "resume" not in request.files:
        return jsonify({"error": "No resume file uploaded"}), 400
    if not request.form.get("jd"):
        return jsonify({"error": "No job description provided"}), 400

    file = request.files["resume"]
    jd = request.form.get("jd")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        import pymupdf
        from pymupdf_rag import to_markdown
        with pymupdf.open(tmp_path) as doc:
            resume_text = to_markdown(doc, pages=range(doc.page_count))

        prompt = JD_MATCH_PROMPT.replace("{jd}", jd).replace("{resume}", resume_text)
        result_text = call_bedrock(prompt).strip()

        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(result_text)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error matching resume: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
