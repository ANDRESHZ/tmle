from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    name='tmle',
    version='0.1',
    author='Lukasz Ambroziak'
    description='Transfer learning'
    long_description=open('README.md').read(),
    url='https://github.com/stasulam/tmle',
    install_requires=requirements,
    license='MIT',
    packages=['tmle'],
    zip_safe=True
)