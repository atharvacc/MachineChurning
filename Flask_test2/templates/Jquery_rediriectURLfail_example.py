

@app.route('/')
def index():
    if not 'username' in session:
        # contains a button showing a login popup form with action set to '/login'
        return render_template('welcome.html')
    else:
        # contains a logout button with a href to '/logout'
        return render_template('webapp.html') 


@app.route('/login', methods=['POST'])
def login():
    session['username'] = request.form['username']
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


add <div data-role="page" id="welcome" data-url="{{ url_for('index') }}"> to file_head 
