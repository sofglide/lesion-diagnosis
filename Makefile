PROJECT_NAME = lesion-diagnosis
PROJECT_READABLE_NAME = "Lesion Diganosis"
PYTHON ?= python3.8
SOURCE_FOLDER = src

###############################################################################
# ENV SETUP                                                                   #
###############################################################################

.PHONY: env-create
env-create:
	$(PYTHON) -m venv .venv --prompt lesiondiagnosis
	make env-update
	#
	# Don't forget to activate the environment before proceeding! You can run:
	# source .venv/bin/activate


.PHONY: env-update
env-update:
	bash -c "\
		. .venv/bin/activate && \
		pip install wheel && \
		pip install --upgrade -r requirements.txt \
	"


.PHONY: env-delete
env-delete:
	rm -rf .venv

###############################################################################
# BUILD: linting                                                              #
###############################################################################


.PHONY: clean
clean:
	rm -rf build dist *.egg-info
	find $(SOURCE_FOLDER) -name __pycache__ | xargs rm -rf
	find tests -name __pycache__ | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
	find $(SOURCE_FOLDER) -name '*.pyc' -delete


.PHONY: reformat
reformat:
	isort $(SOURCE_FOLDER) tests
	black $(SOURCE_FOLDER) tests


.PHONY: lint
lint:
	$(PYTHON) -m pycodestyle $(SOURCE_FOLDER) tests
	$(PYTHON) -m isort --check-only $(SOURCE_FOLDER) tests
	$(PYTHON) -m black --check $(SOURCE_FOLDER) tests
	$(PYTHON) -m pylint $(SOURCE_FOLDER)
	PYTHONPATH=$(SOURCE_FOLDER) $(PYTHON) -m pylint --disable=missing-docstring,no-self-use tests
	$(PYTHON) -m mypy $(SOURCE_FOLDER) tests


.PHONY: test tests
test tests:
	PYTHONPATH=$(SOURCE_FOLDER) $(PYTHON) -m pytest tests/


################e###############################################################
# Experimenting                                                                #
################################################################################
.PHONY: download-data
download-data:
	python src/main.py download-data