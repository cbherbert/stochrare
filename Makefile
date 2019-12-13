develop:
	pip install --user -e .

install:
	pip install --user .

tests:
	python tests.py

clean:
	rm -r dist *.egg-info *~ stochrare/*~
