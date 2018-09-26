install:
	python setup.py install --user

tests:
	python tests.py

clean:
	rm -r dist *.egg-info *~ stochpy/*~
