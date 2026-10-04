import unittest

from bosdyn.api import image_pb2

from scope.sensor_diagnostic import inspect_sources


class SensorTests(unittest.TestCase):
    def test_read_only_report_and_failed_source_isolation(self):
        class Client:
            def list_image_sources(self,**kwargs):
                return [image_pb2.ImageSource(name=n,cols=4,rows=3) for n in ("rgb","broken")]
            def get_image(self,requests,**kwargs):
                name = requests[0].image_source_name
                if name=="broken":
                    raise RuntimeError("camera unavailable")
                response = image_pb2.ImageResponse(status=image_pb2.ImageResponse.STATUS_OK)
                response.shot.acquisition_time.seconds = 100
                response.shot.image.CopyFrom(image_pb2.Image(rows=3,cols=4,
                    format=image_pb2.Image.FORMAT_RAW,pixel_format=image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
                    data=bytes(12)))
                return [response]
        report = inspect_sources(Client(),2)
        self.assertEqual(report["mode"],"READ_ONLY")
        self.assertIsNone(report["network_one_way_ms"])
        self.assertFalse(report["alignment_verified"])
        self.assertEqual(report["sources"][0]["status"],"OK")
        self.assertEqual(report["sources"][1]["status"],"FAILED")
        self.assertIsNone(report["sources"][0]["observed_acquisition_hz"])
        self.assertEqual(report["sources"][0]["duplicate_acquisitions"],1)
        self.assertGreater(report["sources"][0]["decode_latency"]["count"],0)
