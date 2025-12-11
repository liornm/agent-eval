from metrics.base import ComparisonExample, Metric

COMPLETENESS = Metric(
    name="Completeness",
    description=(
        "Whether the solution addresses all aspects of the task: updates all affected call sites, "
        "modifies related tests, handles downstream dependencies, and leaves no loose ends."
    ),
    required_artifacts=["git_diff"],
    examples=[
        ComparisonExample(
            task="fix typo in docs/appcontext.rst formatting issue in docs/design.rst",
            better="""\
diff --git a/docs/appcontext.rst b/docs/appcontext.rst
-When a Flask application handles a request, it pushes a requet context
+When a Flask application handles a request, it pushes a request context

-Lifcycle of the Context
+Lifecycle of the Context

diff --git a/docs/design.rst b/docs/design.rst
-passing around global data. :data:`.current_app: can be used to access the
+passing around global data. :data:`.current_app` can be used to access the
 application object without needing to import the app object directly, avoiding
-circular import issues. :data:`.request`, :data:`.session`, and :data`.g` can be
-imported to access the current data for the request, rather than needing to
+circular import issues. :data:`.request`, :data:`.session`, and :data:`.g` can
+be imported to access the current data for the request, rather than needing to""",
            worse="""\
diff --git a/docs/appcontext.rst b/docs/appcontext.rst
-When a Flask application handles a request, it pushes a requet context
+When a Flask application handles a request, it pushes a request context""",
            explanation=(
                "The better agent fixed all typos in both files mentioned: 'requet' -> "
                "'request', 'Lifcycle' -> 'Lifecycle', and multiple RST formatting "
                "issues. The worse agent only fixed one typo and ignored the rest."
            ),
        ),
        ComparisonExample(
            task=(
                "remove the slsa provenance job from the publish workflow and clean up "
                "any references to it"
            ),
            better="""\
diff --git a/.github/workflows/publish.yaml b/.github/workflows/publish.yaml
     runs-on: ubuntu-latest
-    outputs:
-      hash: ${{ steps.hash.outputs.hash }}
     steps:
-      - name: generate hash
-        id: hash
-        run: cd dist && echo "hash=$(sha256sum * | base64 -w0)" >> $GITHUB_OUTPUT
-  provenance:
-    needs: [build]
-    permissions:
-      actions: read
-      id-token: write
-      contents: write
-    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.1.0
-    with:
-      base64-subjects: ${{ needs.build.outputs.hash }}
   create-release:
-    needs: [provenance]
+    needs: [build]
-          *.intoto.jsonl/* artifact/*
+          artifact/*
   publish-pypi:
-    needs: [provenance]
+    needs: [build]

diff --git a/pyproject.toml b/pyproject.toml
-[tool.gha-update]
-tag-only = [
-    "slsa-framework/slsa-github-generator",
-]""",
            worse="""\
diff --git a/.github/workflows/publish.yaml b/.github/workflows/publish.yaml
-  provenance:
-    needs: [build]
-    permissions:
-      actions: read
-      id-token: write
-      contents: write
-    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v2.1.0
-    with:
-      base64-subjects: ${{ needs.build.outputs.hash }}
   create-release:
-    needs: [provenance]
+    needs: [build]""",
            explanation=(
                "The better agent removed the provenance job AND cleaned up all "
                "references: the hash output, hash generation step, intoto artifact "
                "glob, dependency updates, and the gha-update config in pyproject.toml. "
                "The worse agent only removed the job itself, leaving broken references."
            ),
        ),
    ],
)
