import enum


class Language(str, enum.Enum):
    CSharp = "c_sharp"
    Python = "python"
    Java = "java"
    Kotlin = "kotlin"
    JavaScript = "javascript"
    TypeScript = "typescript"
    PHP = "php"
    Ruby = "ruby"
    C = "c"
    CPP = "cpp"
    Go = "go"
    Swift = "swift"
    Rust = "rust"
    Lua = "lua"
    Bash = "bash"
    R = "r"

    def __str__(self):
        return self.value
