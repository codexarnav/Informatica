import google.generativeai as genai
from transformers import pipeline
api_key=genai.configure(api_key='AIzaSyDVigM7jOug5T5Yb8NoarG9dDA_E47JZ_Q')
model=genai.GenerativeModel('gemini-2.0-flash')


def marketing_content(product_description, target_market, language, usage_context, num_variants):

    format_template = {
        "mail": f"[subject of mail; full mail body content (text in language {language})]",
        "instagram_post": f"[[;-semi colon list of image descriptions]; full caption (text in language {language})]",
        "tweet": f"[[,-semi colon list of image descriptions]; text content (text in language {language})]"
    }

    format_spec = format_template.get(usage_context, "")


    prompt = f"""Create a marketing {usage_context} for a product targeting {target_market}.
    Product Description: {product_description}.
    Answer Format (strictly follow it, use `;` as a separator): {format_spec}"""

    responses = []
    for _ in range(num_variants):
        response = model.generate_content(prompt)
        responses.append(response.text.strip())

    return responses
