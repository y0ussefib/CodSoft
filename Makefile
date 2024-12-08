install:
	pip install -r requirements.txt

run:
	@command -v python3 >/dev/null 2>&1 && python3 main.py || python main.py  # Lancer le jeu


