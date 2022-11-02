from distutils.core import setup
setup(
  name = 'truncatedMVN',
  packages = ['truncatedMVN'],
  version = '1.0',
  description = 'Sample from a truncated MVN',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/truncatedMVN.git',
  download_url = 'https://github.com/lionfish0/truncatedMVN.git',
  keywords = ['gaussian','mvn','truncated','sample','statistics'],
  classifiers = [],
  install_requires=['numpy','scipy'],
)
