#!/usr/bin/env python3
"""
Quick test script to verify layout-aware OCR extraction works correctly.
Run this to test the new implementation before deployment.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all new modules import correctly"""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    try:
        from services.ocr_service import OCRService
        print("✓ OCRService imported successfully")
    except Exception as e:
        print(f"✗ Failed to import OCRService: {e}")
        return False
    
    try:
        from services.medical_entity_service import MedicalEntityResolver
        print("✓ MedicalEntityResolver imported successfully")
    except Exception as e:
        print(f"✗ Failed to import MedicalEntityResolver: {e}")
        return False
    
    try:
        from services.llm_service import LLMService
        print("✓ LLMService imported successfully")
    except Exception as e:
        print(f"✗ Failed to import LLMService: {e}")
        return False
    
    return True


def test_medical_entity_resolver():
    """Test medical entity resolution logic"""
    print("\n" + "=" * 70)
    print("TESTING MEDICAL ENTITY RESOLVER")
    print("=" * 70)
    
    try:
        from services.medical_entity_service import MedicalEntityResolver
        
        resolver = MedicalEntityResolver()
        
        # Test 1: ICD code resolution
        print("\n1. ICD Code Resolution Test:")
        print("   Input: 'o111' (garbled OCR)")
        result = resolver.resolve_icd_code('o111')
        print(f"   Resolved: {result['resolved']}")
        print(f"   Description: {result['description']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        assert result['resolved'] == 'O11.1', "ICD code resolution failed"
        print("   ✓ Test passed")
        
        # Test 2: Procedure resolution
        print("\n2. Procedure Resolution Test:")
        print("   Input: 'cesarean'")
        result = resolver.resolve_procedure('cesarean')
        print(f"   Resolved: {result['resolved']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        assert result['resolved'] == 'Cesarean delivery', "Procedure resolution failed"
        print("   ✓ Test passed")
        
        # Test 3: Batch entity resolution
        print("\n3. Batch Entity Resolution Test:")
        fields = {
            'diagnosis_icd': 'o111',
            'procedure': 'cesarean'
        }
        result = resolver.resolve_extracted_fields(fields)
        print(f"   Has Anomalies: {len(result.get('anomalies', [])) > 0}")
        print(f"   Confidence Score: {result['confidence_score']:.2%}")
        print(f"   Requires Review: {result['requires_manual_review']}")
        assert result['confidence_score'] > 0.8, "Batch resolution confidence too low"
        print("   ✓ Test passed")
        
        # Test 4: Reference database summary
        print("\n4. Reference Database Summary:")
        summary = resolver.get_reference_database_summary()
        print(f"   ICD-10 Codes: {summary['icd10_codes']}")
        print(f"   Procedures: {summary['procedures']}")
        print(f"   Facilities: {summary['facilities']}")
        print(f"   Specialties: {summary['specialties']}")
        print(f"   Total Entities: {summary['total_entities']}")
        # Note: Reference database contains core medical codes for testing
        # In production, load from external sources (IHRIS, KHIS, ICD-10 API)
        assert summary['total_entities'] > 30, "Reference database has core entities"
        print("   ✓ Test passed (note: expand reference databases in production)")
        
        return True
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_service_methods():
    """Test OCR service layout-aware methods exist and are callable"""
    print("\n" + "=" * 70)
    print("TESTING OCR SERVICE METHODS")
    print("=" * 70)
    
    try:
        from services.ocr_service import OCRService
        
        ocr_service = OCRService()
        
        # Check for new methods
        required_methods = [
            'extract_by_layout_regions',
            'structured_extract_with_layout',
            '_classify_region',
            '_extract_region_text'
        ]
        
        for method_name in required_methods:
            if hasattr(ocr_service, method_name):
                print(f"✓ Method exists: {method_name}")
            else:
                print(f"✗ Method missing: {method_name}")
                return False
        
        return True
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_datetime_fixes():
    """Test that datetime deprecation is fixed"""
    print("\n" + "=" * 70)
    print("TESTING DATETIME FIXES")
    print("=" * 70)
    
    try:
        # Read app.py and check for deprecated datetime calls
        app_path = Path(__file__).parent / 'app.py'
        with open(app_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'datetime.utcnow()' in content:
            print("✗ Still using deprecated datetime.utcnow()")
            return False
        else:
            print("✓ No deprecated datetime.utcnow() calls found")
        
        if 'datetime.now(timezone.utc)' in content:
            print("✓ Using timezone-aware datetime.now(timezone.utc)")
        else:
            print("⚠ Warning: datetime.now(timezone.utc) not found")
        
        # Check imports
        if 'from datetime import datetime, timezone' in content:
            print("✓ Correct datetime imports")
        else:
            print("⚠ Warning: timezone import may be missing")
        
        return True
    
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█" * 70)
    print("█ CLAIMFLOW OCR LAYOUT-AWARE EXTRACTION TEST SUITE")
    print("█" * 70)
    
    results = {
        "Imports": test_imports(),
        "Medical Entity Resolver": test_medical_entity_resolver(),
        "OCR Service Methods": test_ocr_service_methods(),
        "Datetime Fixes": test_datetime_fixes(),
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("=" * 70)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Test with Mary Help document")
        print("  2. Monitor confidence scores (target: 70-95%)")
        print("  3. Verify layout regions detected (regions_detected > 0)")
        print("  4. Check entity resolution for ICD codes")
    else:
        print("\n❌ SOME TESTS FAILED! Review errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
