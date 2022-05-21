from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirement_file:
    requirements_list = requirement_file.readlines()
    requirements_list = [lib.replace('\n', '') for lib in requirements_list]

requirements = requirements_list

setup(
    name='Ad Campaign Performance',
    version='0.1.0',
    description='A python implementation of classical, sequential, and machine learning A/B testing to measure brand'
                'awareness',
    url='https://github.com/Yonas-T/ab_testing',
    author='10Academy Batch-5 Group-5',
    author_email='yonaztad@gmail.com, dbulom12@gmail.com, faibagire@gmail.com, diyye101@gmail.com',
    license='MIT License',
    install_requires=requirements,
    long_description=readme,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
