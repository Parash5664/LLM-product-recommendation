# gradio-dashboard.py
import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from db_setup import Base, engine, SessionLocal
from models import User, SearchHistory, Feedback
from sqlalchemy.exc import IntegrityError

# Initialize DB schema
Base.metadata.create_all(bind=engine)

load_dotenv()

# Track the last query
last_query = ""

# Load product dataset
df = pd.read_csv("products_cleaned.csv")
df["large_thumbnail"] = np.where(
    df["imgUrl"].isna() | (df["imgUrl"] == ""),
    "cover-not-found.jpg",
    df["imgUrl"]
)

# Load embedding model and Chroma vectorstore
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_products = Chroma(persist_directory="./chroma_store_test", embedding_function=embedding)

# Log search query
def log_search(username, query):
    db = SessionLocal()
    db.add(SearchHistory(username=username, query=query))
    db.commit()
    db.close()

# Log user feedback
def log_feedback(username, query, rating):
    db = SessionLocal()
    db.add(Feedback(username=username, product_id=query, rating=rating))
    db.commit()
    db.close()
    return f"✅ Feedback '{rating}' recorded for query: {query}"

# User registration
def register_user(username, password):
    db = SessionLocal()
    try:
        db.add(User(username=username, password=password))
        db.commit()
        return True, "✅ Registered successfully. Please log in."
    except IntegrityError:
        db.rollback()
        return False, "❌ Username already exists"
    finally:
        db.close()

# Authenticate user from DB
def authenticate(username, password):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username, User.password == password).first()
    db.close()
    if user:
        return gr.update(visible=True), gr.update(visible=False), ""
    else:
        return gr.update(visible=False), gr.update(visible=True), "❌ Invalid login"

# Retrieve products from vector store
def retrieve_semantic_products(query, top_k=12):
    results = db_products.similarity_search(query, k=top_k)
    matched_ids = [r.metadata.get("id") for r in results if "id" in r.metadata]
    matched_df = df[df["id"].isin(matched_ids)].head(top_k)
    return matched_df

# Recommendation function that returns safe HTML
def recommend_products_with_feedback(query):
    global last_query
    last_query = query
    log_search("admin", query)
    data = retrieve_semantic_products(query)
    if data.empty:
        return "<p>😕 No products found for your query. Try something else.</p>"

    html = "<div style='display: flex; flex-wrap: wrap;'>"
    for _, row in data.iterrows():
        product_id = row["id"]
        img = row["large_thumbnail"]
        text = f"<b>{row['title']}</b><br>$ {row['price']} &nbsp; ⭐ {row['stars']}<br>{row['category_name']}"
        html += f"""
        <div style='width: 22%; margin: 10px; border: 1px solid #ccc; border-radius: 10px; padding: 10px; text-align: center;'>
            <img src='{img}' style='width:100%; height:150px; object-fit:cover;'><br>
            {text}<br><br>
        </div>
        """
    html += "</div>"
    return html

# View search history
def view_search_history():
    db = SessionLocal()
    history = db.query(SearchHistory).order_by(SearchHistory.timestamp.desc()).limit(10).all()
    db.close()
    return "\n".join([f"{h.timestamp} — {h.username}: {h.query}" for h in history])

# View feedback
def view_feedback():
    db = SessionLocal()
    feedbacks = db.query(Feedback).order_by(Feedback.timestamp.desc()).limit(10).all()
    db.close()
    return "\n".join([f"{f.timestamp} — {f.username} rated {f.product_id} as {f.rating}" for f in feedbacks])

# View registered users
def view_registered_users():
    db = SessionLocal()
    users = db.query(User).all()
    db.close()
    return "\n".join([f"🧑 {u.username}" for u in users])

# Gradio UI
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("## 🔐 Login or Register to Access Product Recommendations")

    with gr.Column(visible=True) as login_section:
        user = gr.Textbox(label="Username")
        pwd = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        register_btn = gr.Button("Register")
        login_msg = gr.Textbox(visible=False, interactive=False, label="", show_label=False)

    with gr.Column(visible=False) as app_section:
        gr.Markdown("## 🛍️ AI-Powered Product Recommender")
        with gr.Row():
            query_box = gr.Textbox(label="Search products", placeholder="e.g., noise-cancelling headphones")
            search_btn = gr.Button("Search")
        recommendations = gr.HTML()

        with gr.Row():
            thumbs_up_btn = gr.Button("👍 I liked the recommendations")
            thumbs_down_btn = gr.Button("👎 I didn’t like the recommendations")

        gr.Markdown("### 📜 Search History")
        history_btn = gr.Button("View Recent Searches")
        history_out = gr.Textbox(lines=5, interactive=False)

        gr.Markdown("### 🗳️ Feedback Logs")
        feedback_btn = gr.Button("View Feedback")
        feedback_out = gr.Textbox(lines=5, interactive=False)

        gr.Markdown("### 👥 Registered Users")
        users_btn = gr.Button("View Registered Users")
        users_out = gr.Textbox(lines=5, interactive=False)

    def handle_register(username, password):
        success, message = register_user(username, password)
        if success:
            return gr.update(visible=False), gr.update(visible=True), message
        else:
            return gr.update(visible=True), gr.update(visible=False), message

    login_btn.click(authenticate, [user, pwd], [app_section, login_msg, login_msg])
    register_btn.click(handle_register, [user, pwd], [app_section, login_section, login_msg])
    search_btn.click(lambda query: recommend_products_with_feedback(query), query_box, recommendations)
    thumbs_up_btn.click(lambda: log_feedback("admin", last_query, "up"), outputs=[])
    thumbs_down_btn.click(lambda: log_feedback("admin", last_query, "down"), outputs=[])
    history_btn.click(view_search_history, outputs=history_out)
    feedback_btn.click(view_feedback, outputs=feedback_out)
    users_btn.click(view_registered_users, outputs=users_out)

if __name__ == "__main__":
    dashboard.launch(share=True)
