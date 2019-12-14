"""Helper functions for creating HTML documents by "merging" with a template"""
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import os as _os
import sys as _sys
import shutil as _shutil
import webbrowser as _webbrowser

import re as _re
import subprocess as _subprocess

from pathlib import Path

from ..tools import compattools as _compat
from ..tools import timed_block as _timed_block
from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


def read_contents(filename):
    """
    Read the contents from `filename` as a string.

    Parameters
    ----------
    filename : str

    Returns
    -------
    str
    """
    contents = None
    try:  # on Windows using python3 open can fail when trying to read text files. encoding fixes this
        f = open(filename)
        contents = f.read()
    except UnicodeDecodeError:
        f = open(filename, encoding='utf-8')  # try this, but not available in python 2.7!
        contents = f.read()

    f.close()

    try:  # to convert to unicode since we use unicode literals
        contents = contents.decode('utf-8')
    except AttributeError: pass  # Python3 case when unicode is read in natively (no need to decode)

    return contents


def insert_resource(connected, online_url, offline_filename,
                    integrity=None, crossorigin=None):
    """
    Return the HTML used to insert a resource into a larger HTML file.

    When `connected==True`, an internet connection is assumed and
    `online_url` is used if it's non-None; otherwise `offline_filename` (assumed
    to be relative to the "templates/offline" folder within pyGSTi) is inserted
    inline.  When `connected==False` an offline folder is assumed to be present
    in the same directory as the larger HTML file, and a reference to
    `offline_filename` is inserted.

    Parameters
    ----------
    connected : bool
        Whether an internet connection should be assumed.  If False, then an
        'offline' folder is assumed to be present in the output HTML's folder.

    online_url : str
        The url of the inserted resource as available from the internet.  None
        if there is no such availability.

    offline_filename : str
        The filename of the resource relative to the `templates/offline` pyGSTi
        folder.

    integrity : str, optional
        The "integrity" attribute string of the <script> tag used to reference
        a *.js (javascript) file on the internet.

    crossorigin : str, optional
        The "crossorigin" attribute string of the <script> tag used to reference
        a *.js (javascript) file on the internet.

    Returns
    -------
    str
    """
    if connected:
        if online_url:
            url = online_url
        else:
            #insert resource inline, since we don't want
            # to depend on offline/ directory when connected=True
            assert(offline_filename), \
                "connected=True without `online_url` requires offline filename!"
            absname = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                    "templates", "offline", offline_filename)

            if offline_filename.endswith("js"):
                return '<script type="text/javascript">\n' + \
                    read_contents(absname) + "</script>\n"

            elif offline_filename.endswith("css"):
                return '<style>\n' + read_contents(absname) + "</style>\n"

            else:
                raise ValueError("Unknown resource type for %s" % offline_filename)

    else:
        assert(offline_filename), "connected=False requires offline filename"
        url = "offline/" + offline_filename

    if url.endswith("js"):

        tag = '<script src="%s"' % url
        if connected:
            if integrity: tag += ' integrity="%s"' % integrity
            if crossorigin: tag += ' crossorigin="%s"' % crossorigin
        tag += '></script>'
        return tag

    elif url.endswith("css"):
        return '<link rel="stylesheet" href="%s">' % url

    else:
        raise ValueError("Unknown resource type for %s" % url)


def rsync_offline_dir(outputDir):
    """
    Copy the pyGSTi 'offline' directory into `outputDir` by creating or updating
    any outdated files as needed.
    """
    destDir = _os.path.join(outputDir, "offline")
    offlineDir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "templates", "offline")  # TODO package resources?
    if not _os.path.exists(destDir):
        _shutil.copytree(offlineDir, destDir)

    else:
        for dirpath, _, filenames in _os.walk(offlineDir):
            for nm in filenames:
                srcnm = _os.path.join(dirpath, nm)
                relnm = _os.path.relpath(srcnm, offlineDir)
                destnm = _os.path.join(destDir, relnm)

                if not _os.path.isfile(destnm) or \
                        (_os.path.getmtime(destnm) < _os.path.getmtime(srcnm)):
                    _shutil.copyfile(srcnm, destnm)
                    #print("COPYING to %s" % destnm)


def read_and_preprocess_template(templateFilename, toggles):
    """
    Load a HTML template from a file and perform an preprocessing,
    indicated by "#iftoggle(name)", "#elsetoggle", and "#endtoggle".

    Parameters
    ----------
    templateFilename : str
        filename (no relative directory assumed).

    toggles : dict
        A dictionary of toggle_name:bool pairs specifying
        how to preprocess the template.

    Returns
    -------
    str
    """
    template = read_contents(templateFilename)

    if toggles is None:
        toggles = {}

    def preprocess(txt):
        """ Apply preprocessor directives on `txt` """
        try: i = txt.index("#iftoggle(")
        except ValueError: i = None

        try: k = txt.index("#elsetoggle")
        except ValueError: k = None

        try: j = txt.index("#endtoggle")
        except ValueError: j = None

        if i is None:
            return txt  # no iftoggle, so no further processing to do

        if (k is not None and k < i) or (j is not None and j < i):
            return txt  # else/end appears *before* if - so don't process the if

        #Process the #iftoggle
        off = len("#iftoggle(")
        end = txt[i + off:].index(')')
        toggleName = txt[i + off:i + off + end]
        pre_text = txt[0:i]  # text before our #iftoggle
        post_text = preprocess(txt[i + off + end + 1:])  # text after

        if_text = ""
        else_text = ""

        #Process #elsetoggle or #endtoggle - whichever is first
        try: k = post_text.index("#elsetoggle")  # index in (new) *post_text*
        except ValueError: k = None
        try: j = post_text.index("#endtoggle")  # index in (new) *post_text*
        except ValueError: j = None

        if k is not None and (j is None or k < j):  # if-block ends at #else
            #process #elsetoggle
            if_text = post_text[0:k]
            post_text = preprocess(post_text[k + len("#elsetoggle"):])
            else_processed = True
        else: else_processed = False

        #Process #endtoggle
        try: j = post_text.index("#endtoggle")  # index in (new) *post_text*
        except ValueError: j = None
        assert(j is not None), "#iftoggle(%s) without corresponding #endtoggle" % toggleName

        if not else_processed:  # if-block ends at #endtoggle
            if_text = post_text[0:j]
        else:  # if-block already captured; else-block ends at #endtoggle
            else_text = post_text[0:j]
        post_text = preprocess(post_text[j + len("#endtoggle"):])

        if toggles[toggleName]:
            return pre_text + if_text + post_text
        else:
            return pre_text + else_text + post_text

    return preprocess(template)


def clearDir(path):
    """ If `path` is a directory, remove all the files within it """
    if not _os.path.isdir(path): return
    for fn in _os.listdir(path):
        full_fn = _os.path.join(path, fn)
        if _os.path.isdir(full_fn):
            clearDir(full_fn)
            _os.rmdir(full_fn)
        else:
            _os.remove(full_fn)


def makeEmptyDir(dirname):
    """ Ensure that `dirname` names an empty directory """
    if not _os.path.exists(dirname):
        _os.makedirs(dirname)
    else:
        assert(_os.path.isdir(dirname)), "%s exists but isn't a directory!" % dirname
        clearDir(dirname)
    return dirname


def _render_as_html(value, render_options, link_to):
    """Render an object as HTML.

    See
    ---
    :func:`render_as_html`
    """
    if isinstance(value, str):
        html = value
    else:
        if hasattr(value, 'set_render_options'):
            value.set_render_options(**render_options)

            out = value.render('html')
            if link_to:
                value.set_render_options(leave_includes_src=('tex' in link_to),
                                         render_includes=('pdf' in link_to),
                                         switched_item_mode='separate files')
                if 'tex' in link_to or 'pdf' in link_to: value.render('latex')
                if 'pkl' in link_to: value.render('python')

        else:  # switchboards usually
            out = value.render('html')

        # Note: out is a dictionary of rendered portions
        html = "<script>%(js)s</script>%(html)s" % out

    return html


def render_as_html(qtys, render_options, link_to, verbosity):
    """
    Render the workspace quantities (outputs and switchboards) in the `qtys`
    dictionary as HTML.

    Parameters
    ----------
    qtys : dict
        A dictionary of workspace quantities to render.

    render_options : dict
        a dictionary of render options to set via the
        `WorkspaceOutput.set_render_options` method of workspace output objects.

    link_to : tuple
        If not None, a list of one or more items from the set
        {"tex", "pdf", "pkl"} indicating whether or not to
        create and include links to Latex, PDF, and Python pickle
        files, respectively.

    verbosity : int
        How much detail to print to stdout.

    Returns
    -------
    dict
        With the same keys as `qtys` and values which contain the objects
        rendered as strings.
    """
    printer = _VerbosityPrinter.build_printer(verbosity)

    #render quantities as HTML
    qtys_html = _collections.defaultdict(lambda: "OMITTED")
    for key, val in qtys.items():
        with _timed_block(key, formatStr='Rendering {:35}', printer=printer, verbosity=2):
            qtys[key] = _render_as_html(val, render_options, link_to)

    return qtys_html


def render_as_latex(qtys, render_options, verbosity):
    """
    Render the workspace quantities (outputs; not switchboards) in the `qtys`
    dictionary as LaTeX.

    Parameters
    ----------
    qtys : dict
        A dictionary of workspace quantities to render.

    render_options : dict
        a dictionary of render options to set via the
        `WorkspaceOutput.set_render_options` method of workspace output objects.

    verbosity : int
        How much detail to print to stdout.

    Returns
    -------
    dict
        With the same keys as `qtys` and values which contain the objects
        rendered as strings.
    """
    printer = _VerbosityPrinter.build_printer(verbosity)
    from .workspace import Switchboard as _Switchboard

    #render quantities as Latex
    qtys_latex = _collections.defaultdict(lambda x=0: "OMITTED")
    for key, val in qtys.items():
        if isinstance(val, _Switchboard):
            continue  # silently don't render switchboards in latex
        if _compat.isstr(val):
            qtys_latex[key] = val
        else:
            printer.log("Rendering %s" % key, 3)
            if hasattr(val, 'set_render_options'):
                val.set_render_options(**render_options)
            render_out = val.render("latex")

            # Note: render_out is a dictionary of rendered portions
            qtys_latex[key] = render_out['latex']

    return qtys_latex


def _make_jinja_env(static_path, templateDir=None, render_options=None, link_to=None):
    """Build a jinja2 environment for generating pyGSTi reports"""
    try:
        import jinja2  # import locally since we don't want to require jinja to import pygsti
    except ImportError:
        raise ImportError(("The 'jinja2' optional package is required to create pyGSTi "
                           "reports, and appears to be missing.  Try 'pip install jinja2'."))

    # Indirect access to offline template elements at generation time
    offline_loader = jinja2.PackageLoader('pygsti', 'report/templates/offline/')

    if templateDir is None:
        # Use root packaged templates directory by default
        loader = jinja2.PackageLoader('pygsti', 'report/templates/')

    elif templateDir.startswith('~'):
        # Assume path after ~ is relative to root packaged template directory
        relpath = templateDir[1:]
        loader = jinja2.PackageLoader('pygsti', _os.path.join('report/templates', relpath))

    else:
        # Use path as is.
        loader = jinja2.FileSystemLoader(templateDir)

    # Construct jinja2 environment
    env = jinja2.Environment(
        loader=loader,
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )

    # jinja environment globals
    def jinja_global(fn):
        """Helper decorator for defining functions with global jinja visibility"""
        env.globals[fn.__name__] = fn
        return fn

    def jinja_filter(fn):
        """Helper decorator for defining jinja filters"""
        env.filters[fn.__name__] = fn
        return fn

    @jinja_global
    def static_ref(path):
        """Render an href to a static resource"""
        return static_path / path

    @jinja_global
    def offline_file(filename):
        """Embed an offline file's contents directly in the document, in a script tag."""
        # XXX we gotta find a better way, this is so wild dude
        contents = offline_loader.get_source(env, filename)[0]
        return jinja2.Markup(contents)

    @jinja_filter
    @jinja2.evalcontextfilter
    def render(eval_ctx, value):
        html = _render_as_html(value, render_options or {}, link_to)
        return jinja2.Markup(html) if eval_ctx.autoescape else html

    return env


def merge_jinja_template(qtys, outputFilename, templateDir=None, templateName='main.html',
                         auto_open=False, precision=None, link_to=None, connected=False, toggles=None,
                         renderMath=True, resizable=True, autosize='none', verbosity=0):
    """
    Renders `qtys` and merges them into a single HTML file `outputFilename`.
    This functions parameters are the same as those of :func:`merge_jinja_template_dir`.

    Returns
    -------
    None
    """

    assert(outputFilename.endswith(".html")), "outputFilename should have ended with .html!"
    out_file = Path(outputFilename).absolute()
    out_path = out_file.parent
    static_path = out_path / 'offline'

    #Copy offline directory into position
    if not connected:
        rsync_offline_dir(str(out_path))

    if link_to is not None:
        figDir = out_path / (out_file.stem + '.figures')
        figDir.mkdir(exist_ok=True)
    else:
        figDir = None

    env = _make_jinja_env(static_path.relative_to(out_path), templateDir=templateDir,
                          render_options=dict(switched_item_mode="inline",
                                              global_requirejs=False,
                                              within_report=True,
                                              resizable=resizable, autosize=autosize,
                                              output_dir=figDir, link_to=link_to,
                                              precision=precision),
                          link_to=link_to
                          )

    # Template rendering parameters are the given qtys plus a 'config' dict
    render_params = {
        **qtys,
        'config': {
            **(toggles or {})
            # TODO whatever should be in config namespace
        }
    }

    # Render main page template to output path
    template = env.get_template(templateName)
    with open(outputFilename, 'w') as outfile:
        outfile.write(template.render(render_params))


def merge_jinja_template_dir(qtys, outputDir, templateDir=None, templateName='main.html',
                             auto_open=False, precision=None, link_to=None, connected=False, toggles=None,
                             renderMath=True, resizable=True, autosize='none', verbosity=0):
    """
    Renders `qtys` and merges them into the HTML files under `templateDir`,
    saving the output under `outputDir`.

    Parameters
    ----------
    qtys : dict
        A dictionary of workspace quantities (switchboards and outputs).

    outputDir : str
        The merged-output directory..

    templateDir : str, optional
        The template filename, relative to pyGSTi's `templates` directory.

    auto_open : bool, optional
        Whether the output file should be automatically opened in a web browser.

    precision : int or dict, optional
        The amount of precision to display.  A dictionary with keys
        "polar", "sci", and "normal" can separately specify the
        precision for complex angles, numbers in scientific notation, and
        everything else, respectively.  If an integer is given, it this
        same value is taken for all precision types.  If None, then
        a default is used.
    link_to : list, optional
        If not None, a list of one or more items from the set
        {"tex", "pdf", "pkl"} indicating whether or not to
        create and include links to Latex, PDF, and Python pickle
        files, respectively.

    connected : bool, optional
        Whether an internet connection should be assumed.  If False, then an
        'offline' folder is assumed to be present in the output HTML's folder.

    toggles : dict, optional
        A dictionary of toggle_name:bool pairs specifying
        how to preprocess the template.

    renderMath : bool, optional
        Whether math should be rendered.

    resizable : bool, optional
        Whether figures should be resizable.

    autosize : {'none', 'initial', 'continual'}
        Whether tables and plots should be resized, either initially --
        i.e. just upon first rendering (`"initial"`) -- or whenever
        the browser window is resized (`"continual"`).

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    None
    """

    # Create output directory if it does not already exist
    out_path = Path(outputDir).absolute()
    out_path.mkdir(parents=True, exist_ok=True)
    static_path = out_path / 'offline'

    #Copy offline directory into position
    if not connected:
        rsync_offline_dir(outputDir)

    if link_to is not None:
        figDir = out_path / 'figures'
        figDir.mkdir(exist_ok=True)
    else:
        figDir = None

    env = _make_jinja_env(static_path.relative_to(out_path), templateDir=templateDir,
                          render_options=dict(switched_item_mode="inline",
                                              global_requirejs=False,
                                              within_report=True,
                                              resizable=resizable, autosize=autosize,
                                              output_dir=figDir, link_to=link_to,
                                              precision=precision),
                          link_to=link_to
                          )

    # Template rendering parameters are the given qtys plus a 'config' dict
    render_params = {
        **qtys,
        'config': {
            **(toggles or {})
            # TODO whatever should be in config namespace
        }
    }

    # Render main page template to output path
    template = env.get_template(templateName)
    with open(out_path / templateName, 'w') as outfile:
        outfile.write(template.render(render_params))


def process_call(call):
    """
    Use subprocess to run `call`.

    Parameters
    ----------
    call : list
        A list of exec name and args, e.g. `['ls','-l','myDir']`

    Returns
    -------
    stdout : str
    stderr : str
    return_code : int
    """
    process = _subprocess.Popen(call, stdout=_subprocess.PIPE,
                                stderr=_subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode


def evaluate_call(call, stdout, stderr, returncode, printer):
    """ Run `call` and raise CalledProcessError if exit code > 0 """
    if len(stderr) > 0:
        printer.error(stderr)
    if returncode > 0:
        raise _subprocess.CalledProcessError(returncode, call)


def merge_latex_template(qtys, templateFilename, outputFilename,
                         toggles=None, precision=None, verbosity=0):
    """
    Renders `qtys` and merges them into the LaTeX file `templateFilename`,
    saving the output under `outputFilename`.

    Parameters
    ----------
    qtys : dict
        A dictionary of workspace quantities (outputs).

    templateFilename : str
        The template filename, relative to pyGSTi's `templates` directory.

    outputFilename : str
        The merged-output filename.

    toggles : dict, optional
        A dictionary of toggle_name:bool pairs specifying
        how to preprocess the template.

    precision : int or dict, optional
        The amount of precision to display.  A dictionary with keys
        "polar", "sci", and "normal" can separately specify the
        precision for complex angles, numbers in scientific notation, and
        everything else, respectively.  If an integer is given, it this
        same value is taken for all precision types.  If None, then
        a default is used.

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    None
    """

    printer = _VerbosityPrinter.build_printer(verbosity)
    templateFilename = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                     "templates", templateFilename)
    output_dir = _os.path.dirname(outputFilename)
    output_base = _os.path.splitext(_os.path.basename(outputFilename))[0]

    #render quantities as LaTeX within dir where report will be simplified
    cwd = _os.getcwd()
    if len(output_dir) > 0: _os.chdir(output_dir)
    try:
        fig_dir = output_base + "_files"  # figure directory relative to output_dir
        if not _os.path.isdir(fig_dir):
            _os.mkdir(fig_dir)

        qtys_latex = render_as_latex(qtys, dict(switched_item_mode="inline",
                                                output_dir=fig_dir,
                                                precision=precision), printer)
    finally:
        _os.chdir(cwd)

    if toggles:
        qtys_latex['settoggles'] = ""
        for toggleNm, val in toggles.items():
            qtys_latex['settoggles'] += "\\toggle%s{%s}\n" % \
                (("true" if val else "false"), toggleNm)

    template = ''
    with open(templateFilename, 'r') as templatefile:
        template = templatefile.read()
    template = template.replace("{", "{{").replace("}", "}}")  # double curly braces (for format processing)
    # Replace template field markers with `str.format` fields.
    template = _re.sub(r"\\putfield\{\{([^}]+)\}\}\{\{[^}]*\}\}", "{\\1}", template)

    # Replace str.format fields with values and write to output file
    if _sys.version_info > (3, 0):
        filled_template = template.format_map(qtys_latex)  # need python 3.2+
    else:
        filled_template = template.format(**qtys_latex)  # no nice defaultdict behavior
        filled_template = filled_template.encode('utf-8')  # Python2: need to re-encode for write(...)

    with open(outputFilename, 'w') as outputfile:
        outputfile.write(filled_template)


def compile_latex_report(report_filename, latex_call, printer, auto_open):
    """
    Compile a PDF report from a TeX file. Will compile twice
    automatically.

    Parameters
    ----------
    report_filename : string
        The full file name, which may (but need not) include a ".tex" or ".pdf"
        extension, of the report input tex and output pdf files.

    latex_call : list of string
        List containing the command and flags in the form that
        :function:`subprocess.check_call` uses.

    printer : VerbosityPrinter
        Printer to handle logging.

    Raises
    ------
    subprocess.CalledProcessException
        If the call to the process comiling the PDF returns non-zero exit
        status.

    """
    report_dir = _os.path.dirname(report_filename)
    report_base = _os.path.splitext(_os.path.basename(report_filename))[0]
    texFilename = report_base + ".tex"
    pdfPathname = _os.path.join(report_dir, report_base + ".pdf")
    call = latex_call + [texFilename]

    cwd = _os.getcwd()
    if len(report_dir) > 0:
        _os.chdir(report_dir)

    try:
        #Run latex
        stdout, stderr, returncode = process_call(call)
        evaluate_call(call, stdout, stderr, returncode, printer)
        printer.log("Initial output PDF %s successfully generated." %
                    pdfPathname)
        # We could check if the log file contains "Rerun" in it,
        # but we'll just re-run all the time now
        stdout, stderr, returncode = process_call(call)
        evaluate_call(call, stdout, stderr, returncode, printer)
        printer.log("Final output PDF %s successfully generated. " %
                    pdfPathname + "Cleaning up .aux and .log files.")
        _os.remove(report_base + ".log")
        _os.remove(report_base + ".aux")
    except _subprocess.CalledProcessError as e:
        printer.error("pdflatex returned code %d " % e.returncode
                      + "Check %s.log to see details." % report_base)
    finally:
        _os.chdir(cwd)

    if auto_open:
        url = 'file://' + _os.path.abspath(pdfPathname)
        printer.log("Opening %s..." % pdfPathname)
        _webbrowser.open(url)


def to_pdfinfo(list_of_keyval_tuples):
    """
    Convert a list of (key,value) pairs to a string in the format expected
    for a latex document's "pdfinfo" directive (for setting PDF file meta
    information).

    Parameters
    ----------
    list_of_keyval_tuples : list
        A list of (key,value) tuples.

    Returns
    -------
    str
    """
    def sanitize(val):
        if type(val) in (list, tuple):
            sanitized_val = "[" + ", ".join([sanitize(el)
                                             for el in val]) + "]"
        elif type(val) in (dict, _collections.OrderedDict):
            sanitized_val = "Dict[" + \
                ", ".join(["%s: %s" % (sanitize(k), sanitize(v)) for k, v
                            in val.items()]) + "]"
        else:
            sanitized_val = sanitize_str(str(val))
        return sanitized_val

    def sanitize_str(s):
        ret = s.replace("^", "")
        ret = ret.replace("(", "[")
        ret = ret.replace(")", "]")
        return ret

    def sanitize_key(s):
        #More stringent string replacement for keys
        ret = s.replace(" ", "_")
        ret = ret.replace("^", "")
        ret = ret.replace(",", "_")
        ret = ret.replace("(", "[")
        ret = ret.replace(")", "]")
        return ret

    sanitized_list = []
    for key, val in list_of_keyval_tuples:
        sanitized_key = sanitize_key(key)
        sanitized_val = sanitize(val)
        sanitized_list.append((sanitized_key, sanitized_val))

    return ",\n".join(["%s={%s}" % (key, val) for key, val in sanitized_list])
