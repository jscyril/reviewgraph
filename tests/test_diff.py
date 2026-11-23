from reviewer.diff import parse_diff

DIFF = """\
diff --git a/app.py b/app.py
index 111..222 100644
--- a/app.py
+++ b/app.py
@@ -10,4 +10,5 @@ def handler():
     a = 1
-    b = 2
+    b = 3
+    c = 4

     return a
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
diff --git a/logo.png b/logo.png
Binary files a/logo.png and b/logo.png differ
diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+--- not a header, just an added line starting with dashes
+print("hi")
\\ No newline at end of file
"""


def test_parse_multi_file_diff():
    files = parse_diff(DIFF)
    assert set(files) == {"app.py", "new.py"}  # deletion and binary have nothing to comment on

    app = files["app.py"]
    # context a=10, b=11 (+), c=12 (+), blank context 13 (space stripped), return 14
    assert app["lines"] == [10, 11, 12, 13, 14]
    assert (app["added"], app["removed"]) == (2, 1)
    assert "   12 +    c = 4" in app["numbered"]

    new = files["new.py"]
    assert new["lines"] == [1, 2]
    assert new["diff"].startswith("diff --git a/new.py")


def test_plain_unified_diff_without_git_header():
    d = "--- x.py\t2024-01-01\n+++ x.py\t2024-01-02\n@@ -1 +1 @@\n-a\n+b\n"
    assert parse_diff(d)["x.py"]["lines"] == [1]
