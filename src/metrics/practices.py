from metrics.base import ComparisonExample, Metric

CODEBASE_PRACTICES = Metric(
    name="Codebase Practices",
    description=(
        "Whether the agent follows the conventions of this codebase: naming conventions, "
        "file organization, error handling patterns, logging practices, and code style."
    ),
    required_artifacts=["git_diff"],
    examples=[
        ComparisonExample(
            task=(
                "fix bug in test client when using follow_redirects the session shows "
                "the state from the first request instead of the last one. think the "
                "contexts are being restored in the wrong order after redirects. update "
                "test_redirect_keep_session to verify the fix and add changelog entry"
            ),
            better="""\
diff --git a/CHANGES.rst b/CHANGES.rst
@@ -3,6 +3,10 @@ Version 3.1.2

 Unreleased

+-   Restore preserved test client contexts in the correct order after
+    redirects so the session reflects the final request when
+    ``follow_redirects`` is used.

diff --git a/src/flask/testing.py b/src/flask/testing.py
@@ -240,9 +240,9 @@ class FlaskClient(Client):
         # Re-push contexts that were preserved during the request.
-        while self._new_contexts:
-            cm = self._new_contexts.pop()
+        for cm in self._new_contexts:
             self._context_stack.enter_context(cm)
+        self._new_contexts.clear()

diff --git a/tests/test_testing.py b/tests/test_testing.py
@@ -142,10 +142,16 @@ def test_redirect_keep_session(app, client, app_ctx):
     @app.route("/", methods=["GET", "POST"])
     def index():
         if flask.request.method == "POST":
-            return flask.redirect("/getsession")
-        flask.session["data"] = "foo"
+            flask.session["data"] = "from-post"
+            return flask.redirect("/setlater")
+        flask.session["data"] = "from-get"
         return "index"

+    @app.route("/setlater")
+    def set_later():
+        flask.session["data"] = "from-redirect"
+        return flask.redirect("/getsession")
+
-        assert flask.session.get("data") == "foo"
+        assert flask.session.get("data") == "from-redirect"

         rv = client.post("/", data={}, follow_redirects=True)
-        assert rv.data == b"foo"
+        assert rv.data == b"from-redirect\"""",
            worse="""\
diff --git a/src/flask/testing.py b/src/flask/testing.py
@@ -240,9 +240,9 @@ class FlaskClient(Client):
-        # Re-push contexts that were preserved during the request.
-        while self._new_contexts:
-            cm = self._new_contexts.pop()
-            self._context_stack.enter_context(cm)
+        # fix: iterate in order instead of popping
+        for cm in self._new_contexts: self._context_stack.enter_context(cm)
+        self._new_contexts.clear()""",
            explanation=(
                "The better agent follows project conventions: adds a changelog entry "
                "under 'Unreleased' in the established format, updates the existing test "
                "to verify the fix, and maintains the code style. The worse agent skips "
                "the changelog, doesn't update tests, uses non-standard comment style, "
                "and puts multiple statements on one line against the project's style."
            ),
        ),
    ],
)
