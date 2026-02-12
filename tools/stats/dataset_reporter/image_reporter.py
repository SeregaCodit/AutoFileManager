import pandas as pd

from const_utils.stats_constansts import ImageStatsKeys
from tools.stats.dataset_reporter.base_reporter import BaseDatasetReporter


class ImageDatasetReporter(BaseDatasetReporter):

    def show_console_report(self, df: pd.DataFrame, target_format: str) -> None:
        total_objects = len(df)
        total_annotations = df[ImageStatsKeys.im_path].nunique()
        objects_per_image = df.groupby(ImageStatsKeys.im_path).size()
        average_density = objects_per_image.mean()
        std_density = objects_per_image.std()

        report_rows = [
            "\n" + self.line,
            f"DATASET REPORT FOR {target_format.upper()}",
            self.line,
            f"Total images             :  {total_annotations}",
            f"Total annotated objects  :  {total_objects}",
            f"Total annotated objects  :  {total_objects}",
            f"Objects per image (avg)  :  {average_density:.2f}",
            f"Density deviation (std)  :  {std_density:.2f}"
        ]

        for section in self.settings.img_dataset_report_schema:
            if "IMAGE QUALITY" in section["title"]:
                report_rows.extend(self._render_section(df, section, total_objects))

        for object_name in sorted(df['class_name'].unique()):
            cls_df = df[df['class_name'] == object_name]
            cls_count = len(cls_df)

            report_rows.append(f"\n >>> CLASS: {object_name} ({(cls_count / total_objects) * 100:.1f}%)")

            for section in self.settings.img_dataset_report_schema:
                report_rows.extend(self._render_section(cls_df, section, cls_count))

            report_rows.append(self.line + "\n")
        self.logger.info("\n".join(report_rows))



