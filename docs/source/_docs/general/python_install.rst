.. _install_python:

Installing Python
=================

Windows
-------

Creating a new project
^^^^^^^^^^^^^^^^^^^^^^

#. To install Python on Windows, go to the Microsoft App Store 
   (for example, `Python 3.10 <https://apps.microsoft.com/detail/9pjpw5ldxlz5?hl=en-US&gl=US>`_).

#. Then open PowerShell. If this is not installed, install is also using the 
   Microsoft App Store (`PowerShell <https://apps.microsoft.com/detail/9mz1snwt0n5d?hl=en-us&gl=US>`_).

#. Then go to the folder where you want to create your code. Use the command *cd* to
   change directory. Replace the <> with the Windows path.

   .. code-block:: console

      $ cd <path to your project>

#. Validate that Python is correctly installed. This command shows you the version number
   of the actively installed Python Interpreter.

   .. code-block:: console

      $ python3 --version

#. Then create a virtual environment. See the `Official Python documentation <https://docs.python.org/3/library/venv.html>`_
   for more information. The command below creates a folder called "venv" which acts
   as your virtual environment.

   .. code-block:: console

      $ python3 -m venv venv

Every time, to activate the project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Go to the folder where your project is located if not already. You should
   have a folder called "venv". If not, please see the previous section.

   .. code-block:: console

      $ cd <path to your project>

#. Then activate the virtual environment by running

  .. code-block:: console

      $ venv/Scripts/Activate.ps1

#. You can now run commands such as

  .. code-block:: console

      $ pip install pyoptex

  to install a package, or 

  .. code-block:: console

      $ python test.py

  to run the Python script called "test.py" (which is located in your project folder).


Linux
-----

The steps are fairly similar to the Windows installation.

Creating a new project
^^^^^^^^^^^^^^^^^^^^^^

#. To install Python on Linux (Ubuntu) follow this `guide <https://www.geeksforgeeks.org/how-to-install-python-in-ubuntu/>`_.

#. Open a terminal (Ctrl+Alt+T)

#. Then go to the folder where you want to create your code. Use the command *cd* to
   change directory. Replace the <> with the Linux path.

   .. code-block:: console

      $ cd <path to your project>

#. Validate that Python is correctly installed. This command shows you the version number
   of the actively installed Python Interpreter.

   .. code-block:: console

      $ python3 --version

#. Then create a virtual environment. See the `Official Python documentation <https://docs.python.org/3/library/venv.html>`_
   for more information. The command below creates a folder called "venv" which acts
   as your virtual environment.

   .. code-block:: console

      $ python3 -m venv venv

Every time, to activate the project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Go to the folder where your project is located if not already. You should
   have a folder called "venv". If not, please see the previous section.

   .. code-block:: console

      $ cd <path to your project>

#. Then activate the virtual environment by running

  .. code-block:: console

      $ source venv/bin/activate

#. You can now run commands such as

  .. code-block:: console

      $ pip install pyoptex

  to install a package, or 

  .. code-block:: console

      $ python test.py

  to run the Python script called "test.py" (which is located in your project folder).
