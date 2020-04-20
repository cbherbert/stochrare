develop:
	pip install --user -e .

install:
	pip install --user .

tests:
	python -m unittest discover

clean:
	rm -r dist *.egg-info *~ stochrare/*~
