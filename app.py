"""Flask entry point for the RCM ear-training app."""

from flask import Flask, render_template_string, request, url_for

from rcm_ear_training.config import STATIC_FOLDER
from rcm_ear_training.questions import create_question
from rcm_ear_training.theory import LEVEL_INTERVALS, get_level_description


app = Flask(__name__, static_folder=str(STATIC_FOLDER))


HOME_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RCM Ear Training</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        h1 {
            font-size: 38px;
            margin-bottom: 8px;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .level-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin-top: 24px;
        }

        .level-button {
            display: block;
            padding: 18px;
            border-radius: 14px;
            background: #222;
            color: white;
            text-decoration: none;
            text-align: center;
            font-size: 18px;
        }

        .level-button:hover {
            background: #444;
        }

        .muted {
            color: #666;
        }

        .level-info {
            margin-top: 26px;
            padding: 16px;
            border-radius: 14px;
            background: #f5f5f5;
        }

        .level-info p {
            margin: 8px 0;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>RCM Ear Training</h1>
        <p class="muted">Choose your level to practise interval identification.</p>

        <div class="level-grid">
            {% for level in levels %}
                <a class="level-button" href="/practice/{{ level }}">
                    Level {{ level }}
                </a>
            {% endfor %}
        </div>

        <div class="level-info">
            {% for level in levels %}
                <p><strong>Level {{ level }}:</strong> {{ descriptions[level] }}</p>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""


PRACTICE_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Practice</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        h1 {
            margin-bottom: 8px;
        }

        audio {
            width: 100%;
            margin: 22px 0;
        }

        button {
            display: block;
            width: 100%;
            margin: 10px 0;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #bbb;
            background: white;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #f3f3f3;
        }

        .home-link {
            display: inline-block;
            margin-top: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Level {{ question.level }} Intervals</h1>

        {% if question.level <= 4 %}
            <p>Listen to the interval played up and back down, then choose the interval.</p>
        {% else %}
            <p>Listen to the interval melodically, then harmonically, then choose the interval.</p>
        {% endif %}

        <audio controls>
            <source src="{{ audio_url }}" type="audio/wav">
        </audio>

        <form action="/check" method="post">
            <input type="hidden" name="level" value="{{ question.level }}">
            <input type="hidden" name="correct_answer" value="{{ question.answer }}">

            {% for choice in question.choices %}
                <button type="submit" name="selected_answer" value="{{ choice }}">
                    {{ choice }}
                </button>
            {% endfor %}
        </form>

        <a class="home-link" href="/">Back to levels</a>
    </div>
</body>
</html>
"""


RESULT_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Result</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        .button {
            display: inline-block;
            margin-top: 20px;
            margin-right: 10px;
            padding: 13px 20px;
            border-radius: 12px;
            background: #222;
            color: white;
            text-decoration: none;
            font-size: 16px;
        }

        .secondary {
            background: #666;
        }

        .correct {
            color: green;
        }

        .wrong {
            color: #b00020;
        }
    </style>
</head>
<body>
    <div class="card">
        {% if is_correct %}
            <h1 class="correct">Correct!</h1>
        {% else %}
            <h1 class="wrong">Not quite</h1>
            <p>The answer was {{ correct_answer }}.</p>
        {% endif %}

        <a class="button" href="/practice/{{ level }}">Next Question</a>
        <a class="button secondary" href="/">Choose Level</a>
    </div>
</body>
</html>
"""


ERROR_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Missing Sample</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 920px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            background: #fafafa;
        }

        .card {
            border: 1px solid #ddd;
            border-radius: 18px;
            padding: 30px;
            background: white;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        }

        code {
            background: #f3f3f3;
            padding: 3px 6px;
            border-radius: 6px;
        }

        a {
            display: inline-block;
            margin-top: 20px;
            padding: 13px 20px;
            border-radius: 12px;
            background: #222;
            color: white;
            text-decoration: none;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Missing piano sample</h1>
        <p>{{ error_message }}</p>
        <p>Check that the needed AIFF file exists inside:</p>
        <p><code>static/piano_samples</code></p>
        <a href="/">Back Home</a>
    </div>
</body>
</html>
"""


@app.route("/")
def home():
    levels = sorted(LEVEL_INTERVALS.keys())
    descriptions = {level: get_level_description(level) for level in levels}

    return render_template_string(
        HOME_PAGE,
        levels=levels,
        descriptions=descriptions,
    )


@app.route("/practice/<int:level>")
def practice(level):
    try:
        question = create_question(level)
        audio_url = url_for("static", filename=f"audio/{question.audio_file}")

        return render_template_string(
            PRACTICE_PAGE,
            question=question,
            audio_url=audio_url,
        )

    except (FileNotFoundError, ValueError) as error:
        return render_template_string(ERROR_PAGE, error_message=str(error))


@app.route("/check", methods=["POST"])
def check():
    selected_answer = request.form.get("selected_answer")
    correct_answer = request.form.get("correct_answer")
    level = request.form.get("level")

    return render_template_string(
        RESULT_PAGE,
        is_correct=selected_answer == correct_answer,
        correct_answer=correct_answer,
        level=level,
    )


if __name__ == "__main__":
    app.run(debug=True)
