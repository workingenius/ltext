from distutils.core import setup
setup(
    name='ltext',
    version='0.1.0',
    py_modules=['ltext'],
    author='Nathan Hardy',
    author_email='workingenius@163.com',
    url='https://github.com/workingenius/ltext',
    description='A tool for annotation offset calculation',
    license='LGPLv3',
    extras_require={
        'pretty_print': ['termcolor'],
        'dev': ['coverage'],
    },
)
