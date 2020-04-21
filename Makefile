develop:
	pip install --user -e .

install:
	pip install --user .

tests:
	python -m unittest discover

coverage:
	coverage run -m unittest discover
	coverage report

clean:
	rm -r dist *.egg-info *~ stochrare/*~
	coverage erase
