from setuptools import setup 
import io

setup(	name='mantis_ml',
	packages=['mantis_ml'],	
	entry_points={'console_scripts': ['mantis_ml=mantis_ml.bin.__main__:main']}
					  
)
	
