from flask import Flask, render_template, redirect, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
# from flask_scss import Scss

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


# making a table using sqlalchemy
class MyTask(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(100), nullable=False)
    complete = db.Column(db.Boolean, default=False)
    created = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    def __repr__(self):
        return f'<Task {self.id}>'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        current_task = request.form["content"]
        new_task = MyTask(content=current_task)
        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except Exception as e:
            return f'There was an issue adding your task: {e}'
    elif (request.method == "GET"):
        tasks = MyTask.query.order_by(MyTask.created.desc()).all()
        return render_template('child1.html', tasks=tasks)


@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = MyTask.query.get_or_404(id)
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except Exception as e:
        return f'There was a problem deleting that task: {e}'


@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    task_to_edit = MyTask.query.get_or_404(id)
    if request.method == "POST":
        task_to_edit.content = request.form['content']
        try:
            db.session.commit()
            return redirect('/')
        except Exception as e:
            return f'There was an issue updating your task: {e}'
    else:
        return render_template('edit.html', task=task_to_edit)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)