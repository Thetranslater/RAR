from pathlib import Path

from rar_agent.domain.models import DatasetBundle
from rar_agent.extensions import ReportGenerator


class PlainReport:
    def generate(self, bundle: DatasetBundle, output_dir: Path) -> Path:
        output = output_dir / "report.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        output.write_text(bundle.name, encoding="utf-8")
        return output


def test_report_generator_remains_a_replaceable_extension(tmp_path: Path) -> None:
    generator = PlainReport()
    bundle = DatasetBundle(
        name="Demo", resources=[], characters=[], conversations=[]
    )

    assert isinstance(generator, ReportGenerator)
    assert generator.generate(bundle, tmp_path).read_text(encoding="utf-8") == "Demo"
