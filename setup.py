from setuptools import setup 
import io

setup(	name='mantis_ml',
	packages=['mantis_ml'],	
	entry_points={'console_scripts': ['mantisml=mantis_ml.modules.main.__main__:main',
					  'mantisml-profiler=mantis_ml.modules.profiler.__main__:main',
					  'mantisml-overlap=mantis_ml.modules.hypergeom_enrichment.__main__:main']}
					  
)
	
