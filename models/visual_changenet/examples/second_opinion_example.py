"""Template for plugging the in-context vision model into the disagreement detector.

    python scripts/scene_inspect.py decide --run runs/<run> --second-opinion py:examples/second_opinion_example.py:judge

judge() is called once per fixture position. Replace the body with a call to your model (Cosmos Reason, a VLM
API, a notebook function) and return its answer, or return None to skip a position. Keep test-set images
out of the prompt examples (brief rule).

If your model runs somewhere else, write a CSV instead (scene,position,label,confidence,reason) and use
--second-opinion csv:path/to/answers.csv
"""


def judge(item):
    # item: {"scene", "position", "score", "crop", "golden", "scene_image"}; image paths are absolute
    # Example placeholder: no opinion. Swap in e.g. answer = my_vlm(item["crop"], item["golden"], prompt)
    return None
    # return {"label": "crooked_cap", "confidence": 0.8, "reason": "cap tilted left"}
