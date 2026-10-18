"""
data.py — the benchmark questions.

These are 20 short math word problems with known numeric answers, written for
this project. To use a real benchmark instead, set USE_GSM8K = True
(needs `pip install datasets`), which loads GSM8K from Hugging Face.
"""

USE_GSM8K = False

QUESTIONS = [
    {"q": "Sam has 12 apples. He gives away 5 and then buys 8 more. How many apples does he have now?", "a": 15},
    {"q": "A notebook costs $3 and a pen costs $1.50. How much do 7 notebooks and 2 pens cost in dollars?", "a": 24},
    {"q": "A train travels at 60 miles per hour for 2.5 hours. How many miles does it travel?", "a": 150},
    {"q": "A class has 30 students and 40% of them are girls. How many boys are in the class?", "a": 18},
    {"q": "A recipe needs 3 eggs per cake. Anna wants to bake 4 cakes and has 5 eggs. How many more eggs does she need to buy?", "a": 7},
    {"q": "A rectangle is 8 meters long and 5 meters wide. What is its perimeter in meters?", "a": 26},
    {"q": "Maria earns $15 per hour. She works 6 hours a day for 5 days. How many dollars does she earn in total?", "a": 450},
    {"q": "Tom reads 25 pages a day. His book has 400 pages and he has already read 150. How many more days will it take him to finish?", "a": 10},
    {"q": "A shirt costs $40. It is discounted by 25%, and then a $5 coupon is applied. What is the final price in dollars?", "a": 25},
    {"q": "Three friends split a $72 dinner bill plus a $9 tip equally. How many dollars does each person pay?", "a": 27},
    {"q": "A water tank holds 200 liters and is 35% full. How many more liters are needed to fill it?", "a": 130},
    {"q": "A pack of 6 pencils costs $2. How many dollars do 30 pencils cost?", "a": 10},
    {"q": "A garden has 4 rows of 9 tomato plants and 3 rows of 7 pepper plants. How many plants are there in total?", "a": 57},
    {"q": "Jane is 3 times as old as her son. Her son is 12. How old will Jane be in 5 years?", "a": 41},
    {"q": "A bakery sold 48 muffins in the morning and half as many in the afternoon. Each muffin costs $2. How many dollars did they make?", "a": 144},
    {"q": "A car goes 25 miles per gallon. A trip is 175 miles and gas costs $4 per gallon. How many dollars does the gas cost?", "a": 28},
    {"q": "There are 5 boxes with 24 cookies each. 18 cookies are eaten, and the rest are packed into bags of 6. How many bags are filled?", "a": 17},
    {"q": "The temperature is 8 degrees and drops 3 degrees every hour for 5 hours. What is the temperature now?", "a": -7},
    {"q": "The average of 4 test scores is 85. A fifth test score is 95. What is the new average?", "a": 87},
    {"q": "A store had 120 shirts. It sold one third on Monday and one quarter of the remaining shirts on Tuesday. How many shirts are left?", "a": 60},
]


def load_questions(n=None):
    if USE_GSM8K:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds.select(range(n or 50)):
            ans = row["answer"].split("####")[-1].strip().replace(",", "")
            items.append({"q": row["question"], "a": float(ans)})
        return items
    return QUESTIONS[:n] if n else QUESTIONS
